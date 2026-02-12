"""Custom Inspect AI task definitions.

Add your own eval tasks here. These supplement the built-in tasks from
inspect_evals (MMLU, HumanEval, GSM8K, etc.).

To use a custom task, add it to configs/eval_suites.yaml:

    tasks:
      my_custom_task:
        inspect_path: evals/example_inspect_task.py@my_custom_task
        args: {}

Then run:
    python scripts/run_evals.py --model-id <model> --tasks my_custom_task

See https://inspect.aisi.org.uk/tasks.html for the full task API.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def example_custom_qa() -> Task:
    """Example: simple Q&A eval with hardcoded samples.

    Replace this with your own dataset and scoring logic.
    """
    dataset = MemoryDataset(
        samples=[
            Sample(
                input="What is the capital of France?",
                target="Paris",
            ),
            Sample(
                input="What is the largest planet in our solar system?",
                target="Jupiter",
            ),
        ]
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=match(),
    )
