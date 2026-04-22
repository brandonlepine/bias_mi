"""MedQA dataset loader for C3 generalization evaluation.

Loads MedQA JSONL files and returns a list of standardised item dicts
with ``prompt``, ``answer``, ``question``, and ``options``.

The prompt is formatted as a zero-shot multiple-choice question matching
the BBQ prompt style used elsewhere in the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logging import log


def _format_prompt(question: str, options: dict[str, str]) -> str:
    """Format a MedQA item as a zero-shot MCQ prompt.

    Matches the style: Question text followed by lettered options,
    ending with "Answer:" to elicit the model's next-token prediction.
    """
    lines = [f"Question: {question}"]
    for letter in sorted(options.keys()):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_medqa_items(
    medqa_path: str | Path,
    *,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load MedQA items from a JSONL file or directory.

    Parameters
    ----------
    medqa_path : str or Path
        Path to either:
        - A single ``.jsonl`` file (loaded directly)
        - A directory containing ``{split}.jsonl``
    split : str
        Which split to load if *medqa_path* is a directory.

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``item_idx`` : int
        - ``prompt`` : str (formatted MCQ prompt)
        - ``question`` : str (raw question text)
        - ``answer`` : str (correct letter, e.g. "E")
        - ``answer_text`` : str (correct answer text)
        - ``options`` : dict[str, str] (letter → text)
        - ``letters`` : tuple[str, ...] (available option letters)
        - ``meta_info`` : str
    """
    path = Path(medqa_path)

    if path.is_file() and path.suffix == ".jsonl":
        jsonl_path = path
    elif path.is_dir():
        jsonl_path = path / f"{split}.jsonl"
        if not jsonl_path.exists():
            # Try common alternatives
            for alt in [f"{split}.jsonl", "test.jsonl", "dev.jsonl"]:
                candidate = path / alt
                if candidate.exists():
                    jsonl_path = candidate
                    break
    else:
        raise FileNotFoundError(
            f"MedQA path {medqa_path} is neither a .jsonl file nor a directory "
            f"containing {split}.jsonl"
        )

    if not jsonl_path.exists():
        raise FileNotFoundError(f"MedQA file not found: {jsonl_path}")

    items: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            question = raw.get("question", "")
            options = raw.get("options", {})
            letters = tuple(sorted(options.keys()))

            # Determine correct answer letter
            answer_idx = raw.get("answer_idx")
            if answer_idx and isinstance(answer_idx, str):
                answer_letter = answer_idx.strip().upper()
            elif raw.get("answer") and isinstance(raw["answer"], str):
                # answer field might be the text; find matching letter
                answer_text = raw["answer"]
                answer_letter = ""
                for k, v in options.items():
                    if v == answer_text:
                        answer_letter = k
                        break
                if not answer_letter and len(raw["answer"]) == 1:
                    answer_letter = raw["answer"].upper()
            else:
                answer_letter = ""

            answer_text = options.get(answer_letter, raw.get("answer", ""))

            items.append({
                "item_idx": idx,
                "prompt": _format_prompt(question, options),
                "question": question,
                "answer": answer_letter,
                "answer_text": answer_text,
                "options": options,
                "letters": letters,
                "meta_info": raw.get("meta_info", ""),
            })

    log(f"MedQA: loaded {len(items)} items from {jsonl_path.name}")
    if items:
        log(f"  Sample letters: {items[0]['letters']}")
        log(f"  Sample answer: {items[0]['answer']}")

    return items
