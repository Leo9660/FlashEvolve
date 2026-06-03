<img src="https://github.com/user-attachments/assets/3f90a44e-68ef-4fcd-b743-d21b01bf19b2"
     align="left"
     width="180" />

FlashEvolve is a modular framework for supporting **agent self-evolution algorithms**.
It expresses GEPA, ACE, Meta-Harness, OpenEvolve, AlphaEvolve, Combee, and
related methods using a common set of artifacts, pools, and stage compositions.
Following the approach in our paper, FlashEvolve also provides asynchronous
execution to reduce wall-clock evolution time and improve proposal throughput.

<br clear="left"/>

## Design

```
Algorithm = Artifact × Pool × Stage composition
```

- **4 LLM-heavy Stages** (`Rollout` / `Reflect` / `Propose` / `Evaluate`)
  declare IO type pairs and an async `process(item)`; the runtime owns
  workers, queues, and scheduling.
- **Pool** holds the evolved artifacts and exposes the two non-LLM
  operations the runtime invokes at iteration boundaries:
  `pool.select_parent(strategy)` and `pool.admit(scored)`.
- **Artifact** is what gets evolved — `PromptArtifact` ships; future
  subclasses cover Playbook / HarnessCode / Program / Memory.
- Stages never own queues and never spawn workers. Users never name
  queues. Staleness is a property of pool semantics, not an algorithm
  flag.

## Install

```bash
# core framework (OpenAI-compatible client only)
pip install -e .

# DSPy-backed SignatureSolver (used by examples/gepa)
pip install -e ".[dspy]"

# examples + smoke tests (adds IFBench's rule-checker stack)
pip install -e ".[examples]"

# optional: vLLM, if you serve the model locally
pip install -e ".[server]"
```

## Quick start

Bring up an OpenAI-compatible endpoint (vLLM is the default; any
OpenAI-compatible URL works):

```bash
bash scripts/start_server.sh                       # Qwen3-8B on :8000
bash scripts/start_server.sh 8001 Qwen/Qwen3-14B   # custom port + model
```

Run the toy GEPA demo (5 trivia samples, 3 evolve iterations):

```bash
python -m examples.gepa.run
```

Run the same demo against the OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
python -m examples.gepa.run --provider openai
```


A minimal pipeline reads like the paper figure:

```python
from flashevolve.artifacts import PromptArtifact
from flashevolve.pools import AppendOnlyPool
from flashevolve.runtime import SyncRuntime
from flashevolve.samplers import FullSampler, RandomSampler
from examples.gepa import (
    GEPAReflect, make_gepa_evaluate, make_gepa_propose, make_gepa_rollout,
)

runtime = SyncRuntime(
    pool=AppendOnlyPool(),
    sampler=RandomSampler(trainset, k=3),
    rollout=make_gepa_rollout(agent=my_agent, metric=my_metric,
                              feedback_fn=my_feedback, llm=client),
    reflect=GEPAReflect(),
    propose=make_gepa_propose(llm=client),
    evaluate=make_gepa_evaluate(agent=my_agent, metric=my_metric,
                                llm=client, sampler=FullSampler(valset)),
    initial_artifact=PromptArtifact(text="Respond to the query."),
    budget=10,
)
await runtime.run()
```

See `examples/gepa/datasets.py` for the **dataset convention** (no
`Dataset` class — a dataset is `list[Any]`, schema by the user) and a
real-task loader (`load_ifbench`).

