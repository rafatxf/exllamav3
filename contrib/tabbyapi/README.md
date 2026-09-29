# TabbyAPI patches

Patches for [TabbyAPI](https://github.com/theroyallab/tabbyAPI) used on the server behind the numbers in
[doc/turing.md](../../doc/turing.md). They apply on top of TabbyAPI `816c321` with `git am`:

```sh
cd tabbyAPI && git checkout 816c321 && git am /path/to/exllamav3/contrib/tabbyapi/*.patch
```

| Patch | Needed for | What it does |
|---|---|---|
| `0001` GET `/v1/status` | dashboards (optional) | In-flight jobs (prefill progress, tokens/s), cache statistics and the metrics of finished requests as JSON: the data behind TabbyAPI's console status line, which only renders on an interactive terminal |
| `0002` grammar filters and reasoning | **Gemma 4 with thinking + `json_schema`** | A grammar filter waits for the end of a reasoning block the model opens itself. Gemma 4's prompt ends at the model turn and the model emits `<|channel>thought ... <channel|>`, so the filter used to constrain the first token and the output was broken JSON. Also adds the `qwen38` and `gemma4` sampler presets (each model's `generation_config.json` defaults) |
| `0003` automatic `max_tokens` | draft models with `output_chunking: false` | The automatic `max_tokens` leaves room for the draft window; without it every request with no `max_tokens` failed with "Job requires N pages (only N-1 available)" |

`0002` and `0003` are general fixes, not specific to Turing, and are candidates for upstream TabbyAPI.
