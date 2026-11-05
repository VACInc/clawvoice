import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

MODEL_ID = os.environ.get("STT_MODEL", "nvidia/parakeet-tdt-0.6b-v2")

app = FastAPI(title="Parakeet-TDT STT", version="1.0.0")
model = None


def get_model():
    global model
    if model is None:
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
        model.eval()
    return model


@app.on_event("startup")
def load_on_startup() -> None:
    get_model()


@app.get("/health")
def health() -> dict:
    return {
        "ok": model is not None,
        "model": MODEL_ID,
        "device": "cuda",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


def _to_wav16k_mono(src: str, dst: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-f", "wav", dst],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _extract_text(out) -> str:
    hyp = out[0] if isinstance(out, (list, tuple)) else out
    if hasattr(hyp, "text"):
        return hyp.text
    if isinstance(hyp, (list, tuple)):
        inner = hyp[0]
        return inner.text if hasattr(inner, "text") else str(inner)
    return str(hyp)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form(default=MODEL_ID, alias="model"),
    language: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> JSONResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as raw, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as conv:
        raw.write(await file.read())
        raw.flush()
        _to_wav16k_mono(raw.name, conv.name)
        out = get_model().transcribe([conv.name])
        text = _extract_text(out).strip()
    return JSONResponse({"text": text})

