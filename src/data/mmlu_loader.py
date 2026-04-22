"""MMLU dataset loader for C3 generalization evaluation.

Loads MMLU parquet files (per-subject or aggregated ``all/`` directory)
and returns a list of standardised item dicts with ``prompt``, ``answer``,
``subject``, and ``choices``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import log


INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _format_prompt(question: str, choices: list[str], subject: str) -> str:
    """Format an MMLU item as a zero-shot MCQ prompt."""
    # Clean subject name for display
    subject_display = subject.replace("_", " ").title() if subject else ""
    lines = []
    if subject_display:
        lines.append(f"Subject: {subject_display}")
    lines.append(f"Question: {question}")
    for i, choice in enumerate(choices):
        letter = INDEX_TO_LETTER.get(i, chr(65 + i))
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_mmlu_items(
    mmlu_path: str | Path,
    *,
    split: str = "test",
    subjects: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load MMLU items from parquet files.

    Parameters
    ----------
    mmlu_path : str or Path
        Path to the MMLU dataset directory.  Can be:
        - The root ``datasets/mmlu/`` directory (loads from ``all/`` subdir
          or iterates per-subject dirs)
        - A specific subject directory
        - A single parquet file
    split : str
        Which split to load (``"test"``, ``"dev"``, ``"validation"``).
    subjects : list[str] or None
        If given, only load these subjects.

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``item_idx`` : int (global index)
        - ``prompt`` : str (formatted MCQ prompt)
        - ``question`` : str (raw question text)
        - ``answer`` : str (correct letter, e.g. "B")
        - ``choices`` : list[str] (four option texts)
        - ``subject`` : str (MMLU subject name)
        - ``letters`` : tuple[str, ...] (always ("A","B","C","D"))
    """
    path = Path(mmlu_path)

    if path.is_file() and path.suffix == ".parquet":
        dfs = [pd.read_parquet(path)]
    elif path.is_dir():
        dfs = _load_from_directory(path, split, subjects)
    else:
        raise FileNotFoundError(f"MMLU path not found: {mmlu_path}")

    if not dfs:
        log(f"WARNING: No MMLU data loaded from {mmlu_path}")
        return []

    combined = pd.concat(dfs, ignore_index=True)

    items: list[dict[str, Any]] = []
    for idx, row in combined.iterrows():
        question = str(row.get("question", ""))
        subject = str(row.get("subject", "")).strip()

        # choices can be a numpy array or list
        choices_raw = row.get("choices", [])
        if hasattr(choices_raw, "tolist"):
            choices = choices_raw.tolist()
        else:
            choices = list(choices_raw)

        # answer is an integer index 0-3
        answer_int = row.get("answer", 0)
        if isinstance(answer_int, str):
            answer_letter = answer_int.upper() if len(answer_int) == 1 else ""
        else:
            answer_letter = INDEX_TO_LETTER.get(int(answer_int), "")

        items.append({
            "item_idx": int(idx),
            "prompt": _format_prompt(question, choices, subject),
            "question": question,
            "answer": answer_letter,
            "choices": choices,
            "subject": subject,
            "letters": ("A", "B", "C", "D"),
        })

    log(f"MMLU: loaded {len(items)} items")
    if items:
        n_subjects = len(set(it["subject"] for it in items if it["subject"]))
        log(f"  Subjects: {n_subjects} unique")

    return items


def _load_from_directory(
    base_dir: Path,
    split: str,
    subjects: list[str] | None,
) -> list[pd.DataFrame]:
    """Load parquet files from MMLU directory structure."""
    dfs: list[pd.DataFrame] = []

    # Try the aggregated "all" directory first
    all_dir = base_dir / "all"
    if all_dir.is_dir():
        pattern = f"{split}-*.parquet"
        files = list(all_dir.glob(pattern))
        if files:
            for f in files:
                df = pd.read_parquet(f)
                if subjects:
                    df = df[df["subject"].isin(subjects)]
                dfs.append(df)
            return dfs

    # Fall back to per-subject directories
    split_pattern = f"{split}-*.parquet"
    subject_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name not in ("all", "auxiliary_train")
    )

    for subj_dir in subject_dirs:
        if subjects and subj_dir.name not in subjects:
            continue
        files = list(subj_dir.glob(split_pattern))
        for f in files:
            df = pd.read_parquet(f)
            if "subject" not in df.columns:
                df["subject"] = subj_dir.name
            dfs.append(df)

    return dfs
