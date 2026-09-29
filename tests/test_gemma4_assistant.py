"""Gemma 4 assistant (MTP drafter): greedy speculative decoding against plain decoding, and the graph-captured drafting
round against the per-step loop. Needs the models:

    EXL3_TEST_GEMMA4_TARGET=/path/to/gemma-4-31B-it-exl3 EXL3_TEST_GEMMA4_ASSISTANT=/path/to/gemma-4-31B-it-assistant \\
        python -m pytest tests/test_gemma4_assistant.py
"""
import os
import pytest
import torch

TARGET = os.environ.get("EXL3_TEST_GEMMA4_TARGET")
ASSISTANT = os.environ.get("EXL3_TEST_GEMMA4_ASSISTANT")
pytestmark = pytest.mark.skipif(not (TARGET and ASSISTANT and torch.cuda.is_available()),
                                reason = "set EXL3_TEST_GEMMA4_TARGET and EXL3_TEST_GEMMA4_ASSISTANT")

PROMPTS = [
    "Write a Python function that merges two sorted lists, with a docstring.",
    "¿Por qué el cielo es azul? Responde en dos párrafos.",
    "What is 17 * 23? Explain the steps.",
]
NEW = 96


@pytest.fixture(scope = "module")
def models():
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    cfg = Config.from_directory(TARGET)
    target = Model.from_config(cfg)
    # One recurrent slot: each slot holds the sliding-window layers' K/V ring (~1.3 GB on the 31B)
    cache = Cache(target, max_num_tokens = 4096, layer_type = CacheLayer_quant, k_bits = 4, v_bits = 4, max_batch_size = 1)
    target.load()
    dcfg = Config.from_directory(ASSISTANT)
    draft = Model.from_config(dcfg)
    dcache = Cache(draft, max_num_tokens = 4096, max_batch_size = 1)
    draft.load()
    tok = Tokenizer.from_config(cfg)
    yield target, cache, draft, dcache, tok
    draft.unload()
    target.unload()


def _generate(models, use_draft, ndt = 6):
    from exllamav3 import Generator, Job
    from exllamav3.generator.sampler.presets import ArgmaxSampler
    target, cache, draft, dcache, tok = models
    gen = Generator(model = target, cache = cache, tokenizer = tok,
                    draft_model = draft if use_draft else None, draft_cache = dcache if use_draft else None,
                    num_draft_tokens = ndt if use_draft else None)
    outs = []
    for p in PROMPTS:
        ids = tok.encode(f"<bos><|turn>user\n{p}<turn|>\n<|turn>model\n", add_bos = False, encode_special_tokens = True)
        job = Job(input_ids = ids, max_new_tokens = NEW, sampler = ArgmaxSampler(), stop_conditions = [])
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            gen.iterate()
        outs.append(job.sequences[0].sequence_ids.torch()[0, ids.shape[-1]:].tolist())
    return outs


def test_graph_round_matches_per_step_loop(models):
    os.environ["EXL3_G4A_GRAPH"] = "0"
    try:
        per_step = _generate(models, True)
    finally:
        os.environ.pop("EXL3_G4A_GRAPH", None)
    graph = _generate(models, True)
    assert graph == per_step


def test_speculative_matches_plain_greedy(models):
    # Verification runs the target on 7 rows instead of 1, so a near-tie can resolve differently late in a long
    # generation; the first tokens must match exactly
    plain = _generate(models, False)
    spec = _generate(models, True)
    for a, b in zip(plain, spec):
        assert a[:48] == b[:48]
