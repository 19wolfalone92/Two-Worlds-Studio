"""Independent transcript comparison: small errors can reverse a whole story.

No speech model, network or synthesis is started here. Feed a transcript from
an independent recognizer; forced alignment to the source is NOT that evidence.
ASR differences require listening, not an automatic claim of incorrect speech.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

NEGATIONS = {"не", "нет", "ни", "нельзя", "никогда", "никто", "ничего"}
NUMBERS = set((
    "ноль один одна одно два две три четыре пять шесть семь восемь девять десять "
    "одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать "
    "семнадцать восемнадцать девятнадцать двадцать тридцать сорок пятьдесят "
    "шестьдесят семьдесят восемьдесят девяносто сто двести триста тысяча "
    "тысячи тысяч миллион миллиона миллионов"
).split())


def tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("Transcript must be text")
    return re.findall(r"[а-яa-z0-9]+", text.lower().replace("ё", "е"))


def word_edits(reference: list[str], observed: list[str]) -> tuple[int, list[dict]]:
    """Levenshtein traceback with bounded, one-byte direction storage."""
    n, m = len(reference), len(observed)
    if max(n, m) > 2000:
        raise ValueError("Compare bounded narration sections of at most 2000 words")
    directions = [bytearray(m + 1) for _ in range(n + 1)]
    directions[0][1:] = bytes([2]) * m
    previous = list(range(m + 1))
    for i, left in enumerate(reference, 1):
        current = [i] + [0] * m
        directions[i][0] = 1
        for j, right in enumerate(observed, 1):
            choices = (previous[j - 1] + (left != right), previous[j] + 1, current[j - 1] + 1)
            direction = min(range(3), key=lambda k: choices[k])
            current[j], directions[i][j] = choices[direction], direction
        previous = current
    edits, i, j = [], n, m
    while i or j:
        direction = directions[i][j]
        if direction == 0:
            if reference[i - 1] != observed[j - 1]:
                edits.append({"kind": "substitute", "reference_index": i - 1,
                              "observed_index": j - 1, "expected": reference[i - 1],
                              "observed": observed[j - 1]})
            i, j = i - 1, j - 1
        elif direction == 1:
            edits.append({"kind": "delete", "reference_index": i - 1,
                          "observed_index": j, "expected": reference[i - 1], "observed": ""})
            i -= 1
        else:
            edits.append({"kind": "insert", "reference_index": i,
                          "observed_index": j - 1, "expected": "", "observed": observed[j - 1]})
            j -= 1
    return previous[m], list(reversed(edits))


def audit_transcript(reference: str, recognized: str, *, critical_terms: list[str] = (),
                     review_wer: float = .12) -> dict:
    expected, observed = tokens(reference), tokens(recognized)
    if not expected:
        raise ValueError("A real source transcript is required")
    if not 0 <= review_wer <= 1:
        raise ValueError("WER review threshold must be within 0..1")
    if not isinstance(critical_terms, (list, tuple)) or len(critical_terms) > 100:
        raise ValueError("Critical terms must be a bounded list")
    count, edits = word_edits(expected, observed)
    critical = []
    for edit in edits:
        changed = {edit["expected"], edit["observed"]}
        if changed & NEGATIONS:
            critical.append({"reason": "negation_changed", **edit})
        if changed & NUMBERS or any(any(c.isdigit() for c in token) for token in changed):
            critical.append({"reason": "number_changed", **edit})

    def occurrences(sequence, part):
        return sum(sequence[i:i + len(part)] == part for i in range(len(sequence) - len(part) + 1))

    for term in critical_terms:
        part = tokens(term)
        if not part or not occurrences(expected, part):
            raise ValueError("An authored critical term is absent from the source transcript")
        if occurrences(observed, part) != occurrences(expected, part):
            critical.append({"reason": "authored_term_changed", "term": term})
    wer = count / len(expected)
    review = bool(critical or wer > review_wer)
    return {
        "source_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
        "reference_words": len(expected), "recognized_words": len(observed),
        "word_errors": count, "word_error_rate": round(wer, 4), "edits": edits,
        "critical_differences": critical, "requires_review": review,
        "status": "LISTENING_REVIEW_REQUIRED" if review else "NO_ASR_DIFFERENCES_REQUIRING_REVIEW",
        "source_text_changed": False, "human_listening_performed": False,
        "speech_quality_approved": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--transcript-file", required=True, type=Path)
    parser.add_argument("--critical-term", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit_transcript(args.reference_file.read_text(encoding="utf-8"),
                              args.transcript_file.read_text(encoding="utf-8"),
                              critical_terms=args.critical_term)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "word_error_rate": result["word_error_rate"],
                      "critical_differences": len(result["critical_differences"])}, ensure_ascii=False))
    return 2 if result["requires_review"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
