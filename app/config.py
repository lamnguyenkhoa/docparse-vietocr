"""Settings from the environment (.env in the project root supported; real environment variables win).

`apply_runtime_env()` must run before paddle / paddlex / torch are imported: it points Paddle's model cache into
MODEL_DIR, turns off the model-source network check and stops Paddle from reserving most of the GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


REC_MODELS = ("vgg_transformer", "vgg_seq2seq")


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8126
    token: str = ""
    device: str = "gpu:0"  # paddle style: gpu:N | cpu. VietOCR follows it (cuda:N | cpu) unless rec_device is set.
    rec_device: str = ""  # VietOCR only; empty = same as device. cpu + gpu:0 keeps just VietOCR on the card.
    rec_model: str = "vgg_transformer"
    model_dir: Path = ROOT / "models"
    max_queue: int = 8
    timeout_s: float = 60.0
    max_upload_mb: float = 25.0
    max_mpix: float = 40.0
    layout_min_score: float = 0.5

    @property
    def paddlex_home(self) -> Path:
        return self.model_dir / "paddlex"

    @property
    def vietocr_dir(self) -> Path:
        return self.model_dir / "vietocr"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        model_dir = Path(_str("DOCPARSE_MODEL_DIR", str(ROOT / "models")))
        if not model_dir.is_absolute():
            model_dir = ROOT / model_dir
        s = cls(
            host=_str("DOCPARSE_HOST", cls.host),
            port=_int("DOCPARSE_PORT", cls.port),
            token=os.environ.get("DOCPARSE_TOKEN", "").strip(),
            device=_str("DOCPARSE_DEVICE", cls.device).lower(),
            rec_device=_str("DOCPARSE_REC_DEVICE", "").lower(),
            rec_model=_str("DOCPARSE_REC_MODEL", cls.rec_model),
            model_dir=model_dir,
            max_queue=_int("DOCPARSE_MAX_QUEUE", cls.max_queue),
            timeout_s=_float("DOCPARSE_TIMEOUT_S", cls.timeout_s),
            max_upload_mb=_float("DOCPARSE_MAX_UPLOAD_MB", cls.max_upload_mb),
            max_mpix=_float("DOCPARSE_MAX_MPIX", cls.max_mpix),
            layout_min_score=_float("DOCPARSE_LAYOUT_MIN_SCORE", cls.layout_min_score),
        )
        if s.rec_model not in REC_MODELS:
            raise ValueError(f"DOCPARSE_REC_MODEL must be one of {REC_MODELS}, got {s.rec_model!r}")
        for name, v in (("DOCPARSE_DEVICE", s.device), ("DOCPARSE_REC_DEVICE", s.rec_device_or_default)):
            if not (v == "cpu" or v == "gpu" or v.startswith("gpu:")):
                raise ValueError(f"{name} must be gpu:N or cpu, got {v!r}")
        return s

    @property
    def rec_device_or_default(self) -> str:
        return self.rec_device or self.device

    @property
    def device_label(self) -> str:
        """For logs and /health: "gpu:0", or "cpu, rec gpu:0" when VietOCR runs elsewhere."""
        rec = self.rec_device_or_default
        return self.device if rec == self.device else f"{self.device}, rec {rec}"


def apply_runtime_env(s: Settings) -> None:
    """Environment for paddle/paddlex. Call before importing them."""
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(s.paddlex_home)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # Grow GPU memory on demand: the card is shared with the reranker.
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
