# OpenEvolve HotpotQA Example

This directory contains the FlashEvolve OpenEvolve-style HotpotQA runner.

Loop:

`select parent/context -> propose -> evaluate -> admit`

Run it with:

```bash
python -m examples.openevolve.run_hotpotqa --provider vllm --budget 100
```

OpenAI-compatible usage:

```bash
export OPENAI_API_KEY=...
python -m examples.openevolve.run_hotpotqa --provider openai --model gpt-4o-mini --budget 100
```

Outputs are written to:

`examples/openevolve/runs/hotpotqa_budget{budget}_seed{seed}/`
