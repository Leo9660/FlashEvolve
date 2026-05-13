"""Dataset conventions for FlashEvolve, with a real-task IFBench example.

FlashEvolve has **no ``Dataset`` class**. A "dataset" is just a
``list[Any]``: the framework treats every sample opaquely and passes it
through ``Sampler`` → ``Agent.run`` → ``metric(sample, output)`` →
``feedback_fn(sample, output, score)`` unchanged.

What this means in practice:

1. **Choose a sample schema** (a plain ``dict``, a frozen ``dataclass``,
   even a ``dspy.Example`` — anything). All four user-supplied pieces
   below must agree on this schema; the framework imposes nothing.
2. **Load** your data as a ``list[your_sample_type]``.
3. **Hand the list to a ``Sampler``** — ``RandomSampler(data, k=...)`` for
   per-iteration minibatches, ``FullSampler(data)`` for full-set
   evaluation, or write your own ``Sampler`` for curriculum / stratified
   sampling.

The four user contracts that must agree on the sample schema:

    Agent.run(artifact, samples, *, llm) -> list[AgentResult]
        reads sample fields, returns one AgentResult per sample
    metric(sample, output) -> float
        scores a single (sample, agent-output) pair
    feedback_fn(sample, output, score) -> str
        per-sample diagnostic text consumed by Reflect
    Sampler — produced your ``samples``; you decide its policy

That's the whole contract. Two reference schemas below.

----------------------------------------------------------------------
Reference schema A — trivia (the schema used in ``run.py``)
----------------------------------------------------------------------

A 5-sample toy dataset where each sample is a ``dict`` with two keys.

    TRAINSET = [
        {"q": "What color is the sky on a clear day?", "gold": "blue"},
        ...
    ]

    def trivia_metric(sample, output):
        return 1.0 if sample["gold"].lower() in output.lower() else 0.0

    def trivia_feedback(sample, output, score):
        return f"Q: {sample['q']!r} → answer {output!r} {'matched' if score else 'missed'} gold {sample['gold']!r}."

    class TriviaAgent(Agent):
        async def run(self, artifact, samples, *, llm):
            # Reads sample["q"]; returns list[AgentResult]
            ...

The schema lives entirely in user code — ``q`` and ``gold`` are not
known to the framework.

----------------------------------------------------------------------
Reference schema B — IFBench (a real instruction-following benchmark)
----------------------------------------------------------------------

IFBench (Khatri et al. 2025; jsonl at ``IFBench_{train,test}.jsonl`` in
the upstream GEPA artifact) ships one JSON object per line with the
following keys:

    {
        "prompt": str,                  # the user query
        "instruction_id_list": [str],   # which instruction-following rules to check
        "kwargs": [dict, ...],          # per-rule parameters (parallel to instruction_id_list)
        "key": int,                     # unique id (unused by metric)
    }

The metric (in ``gepa_artifact.benchmarks.IFBench.ifbench_metric``)
takes the agent's response string, runs every rule in
``instruction_id_list`` against it, and returns ``correct_count / total``
in [0, 1]. The feedback string concatenates rule-by-rule pass/fail
diagnostics — exactly the textual signal GEPA's Reflect / Propose stages
need.

Loader (see ``load_ifbench`` below) returns ``list[dict]`` — one dict
per line, no transformation. The user-defined ``Agent`` reads
``sample["prompt"]`` and passes it to the LLM; the user-defined
``metric`` and ``feedback_fn`` read ``sample["instruction_id_list"]``
and ``sample["kwargs"]`` to evaluate the response.

Standard split (matching ``gepa-artifact/.../IFBench/ifbench_data.py``):

    train = train_val_set[300:600]      # 300 samples; sampled by Rollout's RandomSampler
    val   = train_val_set[:300]         # 300 samples; full-set Eval at acceptance time
    test  = IFBench_test.jsonl          # held out for final reporting
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ifbench(
    data_dir: str | Path,
    split: str,
    *,
    train_slice: tuple[int, int] = (300, 600),
    val_slice: tuple[int, int] = (0, 300),
) -> list[dict[str, Any]]:
    """Load one split of IFBench as ``list[dict]``.

    ``data_dir`` should contain ``IFBench_train.jsonl`` and
    ``IFBench_test.jsonl`` (download from the upstream GEPA artifact).

    Splits:
        - ``"train"`` → ``IFBench_train.jsonl[train_slice]`` (default 300:600)
        - ``"val"``   → ``IFBench_train.jsonl[val_slice]``   (default 0:300)
        - ``"test"``  → ``IFBench_test.jsonl`` (the entire file)

    Each returned dict has keys ``prompt``, ``instruction_id_list``,
    ``kwargs``, ``key`` — see the module docstring for the schema.

    Example:
        >>> trainset = load_ifbench("/path/to/IFBench/data", "train")
        >>> valset   = load_ifbench("/path/to/IFBench/data", "val")
        >>> sampler  = RandomSampler(trainset, k=3, seed=0)
        >>> evaluate = make_gepa_evaluate(agent=ifbench_agent,
        ...     metric=ifbench_metric, llm=client, sampler=FullSampler(valset))
    """
    data_dir = Path(data_dir)
    if split == "test":
        path = data_dir / "IFBench_test.jsonl"
        with path.open() as f:
            return [json.loads(line) for line in f]

    path = data_dir / "IFBench_train.jsonl"
    with path.open() as f:
        all_rows = [json.loads(line) for line in f]

    if split == "train":
        lo, hi = train_slice
    elif split == "val":
        lo, hi = val_slice
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")
    return all_rows[lo:hi]


__all__ = ["load_ifbench"]
