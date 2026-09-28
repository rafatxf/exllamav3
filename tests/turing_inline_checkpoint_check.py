#!/usr/bin/env python3
"""Correctness of the inline recurrent checkpoint (prefill of recurrent models): greedy outputs with the checkpoint
taken inside the chunk vs the separate last-page pass, for prompt lengths around page and chunk boundaries, and
prefix reuse restoring each kind of checkpoint vs a cold run. One model load; the switch is flipped at runtime."""
import os, sys, torch
import exllamav3.generator.job as J
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.presets import ArgmaxSampler

model_dir = os.path.expanduser("~/models/Qwen3.8-27B-exl3-4.0bpw")
config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768, layer_type=CacheLayer_quant, k_bits=4, v_bits=4)
model.load(progressbar=False)
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=2048)
text = open(os.path.expanduser("~/bench/corpus/quijote.txt"), encoding="utf-8").read()[300000:]
ids_all = tok.encode(text)[0]

def reset():
    gen.pagetable.reset_page_table()
    if gen.recurrent_cache is not None:
        gen.recurrent_cache.clear()

def run(ids, n=48):
    job = Job(input_ids=ids.unsqueeze(0), max_new_tokens=n, decode_special_tokens=True, sampler=ArgmaxSampler())
    gen.enqueue(job)
    out, cached = [], None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                if "token_ids" in r: out.append(r["token_ids"].flatten())
                if r.get("eos"): cached = r.get("cached_tokens")
    return torch.cat(out).tolist(), cached

ok = True
print("== greedy outputs, inline checkpoint vs separate last-page pass (cold cache)")
for L in (300, 1000, 1791, 2047, 2048, 2049, 2300, 4097, 5000):
    ids = ids_all[:L]
    res = {}
    for mode in (True, False):
        J._inline_recurrent_checkpoint = mode
        reset()
        res[mode], _ = run(ids)
    same = res[True] == res[False]
    first_diff = next((i for i, (a, b) in enumerate(zip(res[True], res[False])) if a != b), None)
    ok &= same or (first_diff is not None and first_diff >= 16)
    print(f"  L {L:5}: identical {same}" + ("" if same else f" (first difference at token {first_diff})"), flush=True)

print("== prefix reuse: P then P+X, checkpoint restored vs cold")
for LP, LX in ((2300, 700), (4500, 900), (3000, 100)):
    P, PX = ids_all[:LP], ids_all[:LP + LX]
    J._inline_recurrent_checkpoint = True
    reset(); cold, cc = run(PX)
    for mode in (True, False):
        J._inline_recurrent_checkpoint = mode
        reset(); run(P, 8)
        warm, wc = run(PX)
        same = warm == cold
        first_diff = next((i for i, (a, b) in enumerate(zip(warm, cold)) if a != b), None)
        ok &= same or (first_diff is not None and first_diff >= 16)
        print(f"  P {LP} + X {LX}, {'inline' if mode else 'two-pass'} checkpoint: cached tokens {wc} (cold {cc}), "
              f"identical to cold {same}" + ("" if same else f" (first difference at token {first_diff})"), flush=True)
print("RESULT", "OK" if ok else "MISMATCH")
