"""Single offline planning/dispatch contract for existing Kokoro and Qwen TTS.

No new synthesis model or paid fallback. The current stable engine modules stay
authoritative; this adapter never silently writes story dialogue or approves it.
"""
from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTES = ROOT / "audio/voice_routes.json"
REGISTRY = ROOT / "audio/voice_registry.json"
CAST = ROOT / "audio/kokoro_cast.json"
SPEAKER = re.compile(r"^[A-Za-z][A-Za-z_0-9]{0,40}$")


@dataclass(frozen=True)
class VoiceRequest:
    speaker: str
    text: str
    language: str
    style: str = "neutral"

    def __post_init__(self):
        if not isinstance(self.speaker, str) or not SPEAKER.fullmatch(self.speaker):
            raise ValueError("Speaker ID must be a safe registry identifier")
        if not isinstance(self.text, str) or not 1 <= len(self.text.strip()) <= 550:
            raise ValueError("Voice preview requires 1..550 explicitly authored characters")
        if self.language not in {"en", "ru"}:
            raise ValueError("Only configured English and Russian voices are available")


class BaseVoiceEngine(ABC):
    @abstractmethod
    def render(self, task: VoiceRequest, profile: dict, output: Path) -> dict:
        """Produce a real WAV, with no silent fallback, only on explicit call."""


class KokoroVoiceEngine(BaseVoiceEngine):
    def render(self, task: VoiceRequest, profile: dict, output: Path) -> dict:
        from studio.kokoro_tts import synthesize
        return synthesize(task.text, profile["voice"], profile["language"], output)


class QwenVoiceEngine(BaseVoiceEngine):
    def render(self, task: VoiceRequest, profile: dict, output: Path) -> dict:
        from studio.russian_qwen_tts import synthesize
        return synthesize(task.text, profile["voice"], output, style=profile["style"])


ENGINES = {"kokoro": KokoroVoiceEngine, "qwen": QwenVoiceEngine}


def route(task: VoiceRequest, *, registry: Path = REGISTRY,
          cast: Path = CAST, routes: Path = ROUTES) -> dict:
    """Resolve voice identity without synthesis, secrets, uploads, or raw text logs."""
    speakers = json.loads(Path(registry).read_text(encoding="utf-8"))["speakers"]
    if task.speaker not in speakers:
        raise ValueError("Speaker has no voice-identity registry entry")
    if task.language == "en":
        from studio.kokoro_tts import load_cast
        english = load_cast(Path(cast))["speakers"].get(task.speaker)
        if english is None:
            raise ValueError("Speaker has no approved English voice route")
        profile = {"engine": "kokoro", "voice": english["voice"],
                   "language": english["language"], "style": "neutral"}
    else:
        from studio.russian_qwen_tts import STYLE_PROMPTS, VOICES
        russian = json.loads(Path(routes).read_text(encoding="utf-8"))
        if russian.get("schema") != 1:
            raise ValueError("Unknown voice route schema")
        selected = russian["russian"].get(task.speaker)
        if not isinstance(selected, dict) or selected.get("voice") not in VOICES:
            raise ValueError("Speaker has no configured Russian voice")
        style = task.style if task.style != "neutral" else selected.get("style", "neutral")
        if style not in STYLE_PROMPTS:
            raise ValueError("Unsupported Russian voice style")
        profile = {"engine": "qwen", "voice": selected["voice"],
                   "language": "ru", "style": style}
    return {
        "speaker": task.speaker, "identity_version": speakers[task.speaker]["version"],
        "engine": profile["engine"], "voice": profile["voice"],
        "language": task.language, "voice_language": profile["language"],
        "style": profile["style"], "text_sha256": hashlib.sha256(task.text.encode()).hexdigest(),
        "text_length": len(task.text.strip()), "review_status": "unreviewed",
        "synthesis_executed": False, "published": False
    }


def execute(task: VoiceRequest, output: Path, *,
            registry: Path = REGISTRY, cast: Path = CAST, routes: Path = ROUTES) -> dict:
    """Only an explicit caller may request synthesis into a private WAV path."""
    plan = route(task, registry=registry, cast=cast, routes=routes)
    dest = Path(output).resolve()
    if dest.suffix.lower() != ".wav" or ROOT.resolve() == dest or ROOT.resolve() in dest.parents:
        raise ValueError("Voice previews must be private WAV files outside the public repository")
    if dest.exists():
        raise FileExistsError("Existing voice output will never be overwritten")
    profile = {
        "voice": plan["voice"], "language": plan["voice_language"],
        "style": plan["style"]
    }
    engine = ENGINES[plan["engine"]]()
    result = engine.render(task, profile, dest)
    return {"engine": plan["engine"], "speaker": task.speaker,
            "result": result, "review_status": "unreviewed", "published": False}
