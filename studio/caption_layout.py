"""Measured Cyrillic captions for the studio's 1080x1920 safe area.

Line breaks change layout only. Source words and measured speech times are
never rewritten, truncated, stretched or assigned invented timings.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

SAFE_LEFT, SAFE_RIGHT = 76, 156
CAPTION_WIDTH = 1080 - SAFE_LEFT - SAFE_RIGHT
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
)


@lru_cache(maxsize=48)
def font(size: int):
    path = next((p for p in FONT_PATHS if Path(p).is_file()), None)
    if path is None:
        raise ValueError("A real Cyrillic font is required for measured captions")
    return ImageFont.truetype(path, size)


def layout(text: str, *, preferred_size: int = 54, minimum_size: int = 44,
           max_width: int = CAPTION_WIDTH) -> dict:
    """Fit one or two balanced lines, or request shorter *timed* cue groups.

    Uses actual font metrics, rather than a guessed character count. A long
    caption cannot silently turn into three lines or lose its final words.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Caption text is missing")
    if not (type(preferred_size) is int and type(minimum_size) is int
            and 16 <= minimum_size <= preferred_size <= 96
            and type(max_width) is int and max_width > 40):
        raise ValueError("Invalid caption layout bounds")
    words = text.split()
    source = " ".join(words)
    for size in range(preferred_size, minimum_size - 1, -1):
        face = font(size)
        width = float(face.getlength(source))
        if width <= max_width:
            return {"lines": [source], "font_size": size, "widths": [width]}
        candidates = []
        for split in range(1, len(words)):
            lines = [" ".join(words[:split]), " ".join(words[split:])]
            widths = [float(face.getlength(line)) for line in lines]
            if max(widths) <= max_width:
                # Keep lines balanced; favor a natural clause boundary when
                # alternatives have a similar visual weight.
                imbalance = abs(widths[0] - widths[1])
                punctuation = 32 if lines[0].endswith((",", ";", ":", ".", "?", "!", "…")) else 0
                candidates.append((imbalance - punctuation, lines, widths))
        if candidates:
            _, lines, widths = min(candidates, key=lambda row: row[0])
            return {"lines": lines, "font_size": size, "widths": widths}
    raise ValueError("Caption cannot fit two readable lines; regroup measured word cues")


def escaped_ass(text: str) -> str:
    return text.replace("\\", "＼").replace("{", "(").replace("}", ")")


def ass_text(text: str, *, preferred_size: int = 54, minimum_size: int = 44) -> str:
    plan = layout(text, preferred_size=preferred_size, minimum_size=minimum_size)
    return (f"{{\\fs{plan['font_size']}\\q2}}" +
            r"\N".join(escaped_ass(line) for line in plan["lines"]))


def audit_cues(cues: list[dict]) -> dict:
    """Reading-load warnings are editorial evidence, never voice approval."""
    findings = []
    for index, cue in enumerate(cues, 1):
        seconds = cue["end"] - cue["start"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Caption has no finite spoken interval")
        plan = layout(cue["text"])
        cps = len(" ".join(cue["text"].split())) / seconds
        issues = []
        if cps > 24:
            issues.append("fast_reading_load")
        if seconds < .45:
            issues.append("brief_caption")
        if seconds > 4:
            issues.append("long_caption_hold")
        if issues:
            findings.append({"cue": index, "characters_per_second": round(cps, 2),
                             "seconds": round(seconds, 3), "issues": issues})
        assert len(plan["lines"]) <= 2
    return {"cues": len(cues), "max_lines": 2, "warnings": findings,
            "speech_times_changed": False, "source_words_changed": False}
