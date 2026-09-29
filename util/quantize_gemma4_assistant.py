import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["EXL3_G4A_GRAPH"] = "0"   # per-step drafting, so every Linear.forward sees the capture dict
import argparse, glob, json, shutil
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.presets import ArgmaxSampler
from exllamav3.modules import Linear
from exllamav3.modules.quant import LinearEXL3
from exllamav3.loader.safetensors_alt import save_file
from exllamav3.util import Timer

"""
Quantize a Gemma 4 assistant (MTP drafter, e.g. google/gemma-4-31B-it-assistant) to EXL3, with Hessians captured from
its own drafting activations: the target (already EXL3) generates from a set of chat prompts while the unquantized
assistant drafts for it, and every assistant linear accumulates its input statistics. The output directory is a
drop-in replacement for the original assistant (same config, quantized linears, norms and scalars as before).

  python util/quantize_gemma4_assistant.py -t TARGET_EXL3_DIR -a ASSISTANT_DIR -o OUT_DIR [-b 6] [-hb 6]
"""

PROMPTS = [
    "Explain how a hash map works, including collision handling, and give a short Python example.",
    "Write a thread-safe LRU cache with TTL in Python, with type hints and docstrings.",
    "¿Cuál es la receta tradicional de la paella valenciana? Explícala paso a paso.",
    "A bag has 5 red, 4 blue and 3 green balls. Three are drawn without replacement. What is the probability that "
    "all three have different colors? Show your reasoning.",
    "Write a short story about a lighthouse keeper who notices the light is flashing a message.",
    "Summarize the causes and consequences of the French Revolution in a few paragraphs.",
    "Write a SQL query that returns the top 3 customers by total order value per country, and explain it.",
    "Translate to French and German: 'The meeting has been moved to Thursday afternoon because of the storm.'",
    "Implement binary search in Rust and in C, and discuss off-by-one pitfalls.",
    "Explica la diferencia entre TCP y UDP y cuándo conviene usar cada uno.",
    "Derive the formula for the sum of a geometric series and give two applications.",
    "Write a bash script that finds the ten largest files under a directory and prints their sizes.",
    "What are the main differences between transformers and recurrent neural networks?",
    "Escribe un correo formal pidiendo un aplazamiento de una entrega de proyecto.",
    "Write a JSON schema for a user profile with name, email, age and a list of addresses.",
    "Solve: if 3x + 7 = 2x - 5, what is x? Then explain the steps to a child.",
]


def calibrate(args):
    device = torch.device(f"cuda:{args.device}")

    # Saved capture: only the assistant is needed
    if args.load_h:
        acfg = Config.from_directory(args.assistant_dir)
        draft = Model.from_config(acfg)
        Cache(draft, max_num_tokens = 256, max_batch_size = 1)
        draft.load(progressbar = True)
        return torch.load(args.load_h, map_location = device, weights_only = False), draft   # our own capture file

    # Target and unquantized assistant
    tcfg = Config.from_directory(args.target_dir)
    target = Model.from_config(tcfg)
    cache = Cache(target, max_num_tokens = args.ctx, layer_type = CacheLayer_quant, k_bits = 4, v_bits = 4, max_batch_size = 1)
    target.load(progressbar = True)
    acfg = Config.from_directory(args.assistant_dir)
    draft = Model.from_config(acfg)
    dcache = Cache(draft, max_num_tokens = args.ctx, max_batch_size = 1)
    draft.load(progressbar = True)
    tok = Tokenizer.from_config(tcfg)
    gen = Generator(model = target, cache = cache, tokenizer = tok, draft_model = draft, draft_cache = dcache,
                    num_draft_tokens = args.ndt)

    # Capture Hessians while drafting
    capture = {}
    fwd = draft.forward
    def capturing_forward(ids, params):
        params["capture"] = capture
        return fwd(ids, params)
    draft.forward = capturing_forward

    import jinja2
    env = jinja2.Environment(trim_blocks = True, lstrip_blocks = True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    tmpl_path = os.path.join(args.target_dir, "chat_template.jinja")
    tmpl = env.from_string(open(tmpl_path).read()) if os.path.exists(tmpl_path) else None

    for think in (False, True):
        for i, p in enumerate(PROMPTS):
            if tmpl is not None:
                text = tmpl.render(messages = [{"role": "user", "content": p}], add_generation_prompt = True,
                                   enable_thinking = think, bos_token = "<bos>")
            else:
                text = draft.default_chat_prompt(p)
            ids = tok.encode(text, add_bos = False, encode_special_tokens = True)
            job = Job(input_ids = ids, max_new_tokens = args.tokens, sampler = ArgmaxSampler(), stop_conditions = [])
            gen.enqueue(job)
            while gen.num_remaining_jobs():
                gen.iterate()
            rows = max((h["count"] for h in capture.values()), default = 0)
            print(f" -- Calibration prompt {i + 1}/{len(PROMPTS)}{' (thinking)' if think else ''}: {rows} rows", flush = True)
    draft.forward = fwd
    if args.save_h:
        torch.save(capture, args.save_h)

    # Free the target before quantizing
    gen = None
    target.unload()
    del target, cache
    torch.cuda.empty_cache()
    return capture, draft


@torch.inference_mode()
def quantize_and_save(args, capture, draft):
    # Quantize
    linears = [m for m in draft if isinstance(m, Linear) and m.qmap and m.device is not None]
    for i, linear in enumerate(linears):
        K = args.head_bits if linear.key == "lm_head" else args.bits
        H_data = capture[linear.qmap]
        quant_args = {
            "seed": i,
            "mul1": True,
            "K": K,
            "devices": [args.device],
            "device_ratios": None,
            "apply_out_scales": None,
        }
        with Timer() as t:
            proxy_err = linear.convert_exl3(H_data, quant_args = quant_args, progress_str = f" -- <step>: {linear.key}")
        assert isinstance(linear.inner, LinearEXL3)
        print(f" -- Quantized: {linear.key:48}  bpw: {K:5.2f}  proxy_err: {proxy_err:8.6f}  [{t.interval:4.2f} s]",
              flush = True)

    # Save: every tensor of the assistant, quantized linears in EXL3 storage
    tensors = {}
    for m in draft:
        if m.device is None:
            continue
        for k, v in m.get_tensors().items():
            tensors[k] = v.contiguous().cpu()
    os.makedirs(args.out_dir, exist_ok = True)
    save_file(tensors, os.path.join(args.out_dir, "model.safetensors"))
    for f in glob.glob(os.path.join(args.assistant_dir, "*")):
        name = os.path.basename(f)
        if name.endswith(".safetensors") or name.endswith(".index.json") or os.path.isdir(f):
            continue
        shutil.copy(f, os.path.join(args.out_dir, name))
    cfg_path = os.path.join(args.out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "quant_method": "exl3",
        "bits": args.bits,
        "head_bits": args.head_bits,
        "codebook": "mul1",
        "calibration": {"source": "drafting activations", "prompts": 2 * len(PROMPTS), "tokens": args.tokens},
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent = 2)
    print(f" -- Wrote {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("-t", "--target_dir", type = str, required = True, help = "Target model (EXL3) directory")
    parser.add_argument("-a", "--assistant_dir", type = str, required = True, help = "Unquantized assistant directory")
    parser.add_argument("-o", "--out_dir", type = str, required = True, help = "Output directory")
    parser.add_argument("-b", "--bits", type = int, default = 6, help = "Bitrate of the layers, default: 6")
    parser.add_argument("-hb", "--head_bits", type = int, default = 6, help = "Bitrate of the output head, default: 6")
    parser.add_argument("-n", "--tokens", type = int, default = 512, help = "Tokens generated per calibration prompt")
    parser.add_argument("--ndt", type = int, default = 6, help = "Draft tokens per round while calibrating")
    parser.add_argument("--ctx", type = int, default = 8192, help = "Cache size while calibrating")
    parser.add_argument("-d", "--device", type = int, default = 0)
    parser.add_argument("--save_h", type = str, default = None, help = "Save the captured Hessians to this file")
    parser.add_argument("--load_h", type = str, default = None, help = "Skip calibration, load Hessians from this file")
    _args = parser.parse_args()
    _capture, _draft = calibrate(_args)
    quantize_and_save(_args, _capture, _draft)
