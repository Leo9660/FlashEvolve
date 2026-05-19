import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    rows: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} samples from {file_path}")
    return rows


def parse_instruction_and_input(all_context: str) -> tuple[str, str]:
    if "Input: " in all_context and "Instruction: " in all_context:
        instruction_part = all_context.split("Input: ")[0].strip()
        instruction_part = instruction_part.split("Instruction: ")[1].strip()
        remaining = all_context.split("Input: ")[1]
        input_text = remaining.split("Answer: ")[0].strip()
        return input_text, instruction_part
    return "", all_context


def parse_context_and_question_formula(all_context: str) -> tuple[str, str]:
    if "Question: " in all_context and ". Answer:" in all_context:
        parts = all_context.split("Question: ", 1)
        instruction_part = parts[0].strip()
        question_text = parts[1].split(". Answer:")[0].strip()
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        question_text += (
            " Your answer should be a plain floating point number, round to the "
            "nearest hundredth if necessary. Do the necessary conversions, for "
            "example 5 million should be 5000000.0. "
        )
        return "", question_text
    return "", all_context


class FinanceDataProcessor:
    def __init__(self, task_name: str):
        if task_name not in {"finer", "formula"}:
            raise ValueError(f"Unknown task: {task_name}")
        self.task_name = task_name

    def process_task_data(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parse_fn = (
            parse_instruction_and_input
            if self.task_name == "finer"
            else parse_context_and_question_formula
        )
        processed: list[dict[str, Any]] = []
        for item in raw_data:
            context = item.get("context", "")
            target = item.get("target", "")
            input_text, question = parse_fn(context)
            processed.append(
                {
                    "context": input_text,
                    "question": question,
                    "target": target,
                    "others": {
                        "original_context": context,
                        "task": self.task_name,
                        "data_source": "finance",
                    },
                }
            )
        return processed

    def _finer_answer_is_correct(
        self, predicted: str, ground_truth: str, return_counts: bool = False
    ) -> bool | tuple[int, int]:
        pred = [val.lower().strip() for val in predicted.split(",")]
        label = [val.lower().strip() for val in ground_truth.split(",")]
        count = 0
        if len(pred) != len(label):
            if len(pred) > len(label):
                pred = pred[: len(label)]
            else:
                pred += [""] * (len(label) - len(pred))
        for prediction, gold in zip(pred, label):
            try:
                gold = eval(gold)
                prediction = eval(prediction.replace(",", "").replace("$", ""))
            except Exception:
                pass
            if prediction == gold:
                count += 1
        score = count / len(pred) if pred else 0.0
        if return_counts:
            return count, len(pred)
        return score == 1.0

    def _formula_answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        try:
            return float(predicted.replace(",", "")) == float(
                ground_truth.replace(",", "")
            )
        except Exception:
            return predicted == ground_truth

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        if self.task_name == "finer":
            return bool(self._finer_answer_is_correct(predicted, ground_truth))
        return self._formula_answer_is_correct(predicted, ground_truth)

    def _evaluate_finer_accuracy(self, out: list[str], target: list[str]) -> float:
        if len(out) != len(target):
            raise ValueError("Prediction and target lengths must match.")
        correct_count = 0
        total_count = 0
        for predicted, gold in zip(out, target):
            correct, total = self._finer_answer_is_correct(
                predicted, gold, return_counts=True
            )
            correct_count += correct
            total_count += total
        return correct_count / total_count if total_count else 0.0

    def _evaluate_formula_accuracy(self, out: list[str], target: list[str]) -> float:
        if len(out) != len(target):
            raise ValueError("Prediction and target lengths must match.")
        correct_count = 0
        for predicted, gold in zip(out, target):
            if self._formula_answer_is_correct(predicted, gold):
                correct_count += 1
        return correct_count / len(out) if out else 0.0

    def evaluate_accuracy(self, out: list[str], target: list[str]) -> float:
        if self.task_name == "finer":
            return self._evaluate_finer_accuracy(out, target)
        return self._evaluate_formula_accuracy(out, target)
