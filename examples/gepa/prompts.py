"""GEPA prompt templates.

Verbatim from the GEPA reference implementation
(``gepa-artifact/.../teleprompt/gepa/instruction_proposal.py``). The
substitution placeholders ``<curr_instructions>`` and
``<inputs_outputs_feedback>`` are filled by ``gepa_propose_render``.
"""

PROPOSE_PROMPT = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""
