# OpenEvolve Examples

These examples run faithful OpenEvolve-style prompt optimization workflows.

Loop:

`select parent/context -> propose -> evaluate -> admit`

Current runners:

- `python -m examples.openevolve.run_ifbench`
- `python -m examples.openevolve.run_hotpotqa`

The IFBench evaluator performs its own cascade sampling on the IFBench train split,
with final reporting on local val/test splits.

The HotpotQA evaluator performs its own cascade sampling on a HotpotQA train subset,
with final reporting on a held-out validation subset.

IFBench expects local data from:

`./gepa-artifact/gepa_artifact/benchmarks/IFBench/data`

Run IFBench with:

```bash
python -m examples.openevolve.run_ifbench --provider vllm --budget 50
```

Run HotpotQA with:

```bash
python -m examples.openevolve.run_hotpotqa --provider vllm --budget 50
```

OpenAI-compatible usage:

```bash
export OPENAI_API_KEY=...
python -m examples.openevolve.run_ifbench --provider openai --model gpt-4o-mini --budget 50
```

Outputs are written to:

`examples/openevolve/runs/ifbench_budget{budget}_seed{seed}/`
