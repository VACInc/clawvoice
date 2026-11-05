from flask import Flask, request, jsonify, send_file, Response
import io
import logging
import os
import re
import subprocess
import tempfile
import time

import perth
from huggingface_hub import snapshot_download
import torch
import torchaudio as ta
import torchaudio.functional as AF
import threading
import numpy as np
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)

PERTH_FALLBACK = getattr(perth, 'PerthImplicitWatermarker', None) is None
if PERTH_FALLBACK:
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

import chatterbox.tts_turbo as turbo_module

_orig_resample = turbo_module.librosa.resample


def _resample_float32(*args, **kwargs):
    out = _orig_resample(*args, **kwargs)
    return out.astype('float32', copy=False) if hasattr(out, 'astype') else out


turbo_module.librosa.resample = _resample_float32
ChatterboxTurboTTS = turbo_module.ChatterboxTurboTTS

try:
    from pronounce import fix_pronunciation  # spoken-name respelling, pronounce.py
except Exception:  # missing helper must not stop TTS
    fix_pronunciation = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if PERTH_FALLBACK:
    logger.warning("PerthImplicitWatermarker unavailable; using DummyWatermarker fallback")

MODEL = None
MODEL_SR = None
TARGET_SR = 48000
VOICE_DIR = os.environ.get("VOICE_DIR", "/app/voices")
# Every <name>.wav in VOICE_DIR is a voice; request it with "voice": "<name>".
VOICE_MAP = {
    p[:-4]: os.path.join(VOICE_DIR, p) for p in sorted(os.listdir(VOICE_DIR)) if p.endswith(".wav")
} if os.path.isdir(VOICE_DIR) else {}
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "myvoice")
CHUNK_THRESHOLD_CHARS = int(os.environ.get("TURBO_CHUNK_THRESHOLD_CHARS", "260"))
MAX_CHUNK_CHARS = int(os.environ.get("TURBO_MAX_CHUNK_CHARS", "180"))
CHUNK_PAUSE_MS = int(os.environ.get("TURBO_CHUNK_PAUSE_MS", "120"))
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,;:—–-])\s+')
SENTENCE_SPLIT_MARKER = "<TURBO_SENTENCE_SPLIT>"


def load_model():
    global MODEL, MODEL_SR
    if MODEL is None:
        logger.info("Loading ChatterboxTurboTTS on CUDA...")
        t0 = time.time()
        local_path = snapshot_download(
            repo_id="ResembleAI/chatterbox-turbo",
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
            token=False,
        )
        MODEL = ChatterboxTurboTTS.from_local(local_path, device="cuda")
        MODEL_SR = MODEL.sr
        logger.info("Chatterbox loaded in %.2fs, sr=%s", time.time() - t0, MODEL_SR)
    return MODEL


def as_float(v, default):
    try:
        return float(v)
    except Exception:
        return default


def normalize_response_format(fmt: str | None) -> str:
    fmt = (fmt or "mp3").lower()
    if fmt in ("mpeg", "mpga"):
        return "mp3"
    if fmt in ("pcm16", "s16le", "raw"):
        return "pcm"
    if fmt in ("flac", "wav", "mp3", "pcm"):
        return fmt
    return "mp3"


def resolve_target_sample_rate(response_format: str, requested_sample_rate: int | None) -> int:
    if requested_sample_rate and requested_sample_rate > 0:
        return requested_sample_rate
    if response_format == "pcm":
        return 24000
    return TARGET_SR


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_with_word_wrap(text: str, max_chunk_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chunk_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word
    chunks.append(current)
    return chunks


def split_long_unit(text: str, max_chunk_chars: int) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= max_chunk_chars:
        return [text]

    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) == 1:
        return split_with_word_wrap(text, max_chunk_chars)

    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_with_word_wrap(part, max_chunk_chars))
            continue

        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= max_chunk_chars:
            current = candidate
        else:
            chunks.append(current)
            current = part

    if current:
        chunks.append(current)
    return chunks


def split_sentenceish(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    marked = re.sub(
        r'([.!?]["\'\)\]]+)\s+',
        rf'\1{SENTENCE_SPLIT_MARKER}',
        text,
    )
    marked = re.sub(r'([.!?])\s+', rf'\1{SENTENCE_SPLIT_MARKER}', marked)
    return [s.strip() for s in marked.split(SENTENCE_SPLIT_MARKER) if s.strip()]


def split_text_for_turbo(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= CHUNK_THRESHOLD_CHARS:
        return [text]

    sentences = split_sentenceish(text)
    if len(sentences) <= 1:
        return split_long_unit(text, MAX_CHUNK_CHARS)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_unit(sentence, MAX_CHUNK_CHARS))
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks or [text]


def generate_single_chunk(model, text: str, audio_prompt_path: str, exaggeration: float, cfg_weight: float, temperature: float):
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav.cpu()


def generate_with_chunking(model, text: str, audio_prompt_path: str, exaggeration: float, cfg_weight: float, temperature: float):
    chunks = split_text_for_turbo(text)
    if len(chunks) <= 1:
        return generate_single_chunk(model, text, audio_prompt_path, exaggeration, cfg_weight, temperature), chunks

    logger.info(
        "Chunking long request into %d parts (threshold=%d max_chunk=%d pause_ms=%d)",
        len(chunks), CHUNK_THRESHOLD_CHARS, MAX_CHUNK_CHARS, CHUNK_PAUSE_MS,
    )

    def trim_trailing_silence(wav_tensor, sr, threshold=0.005, max_trim_ms=300):
        """Trim trailing silence from a chunk tensor (1, samples)."""
        max_trim = int(sr * max_trim_ms / 1000)
        samples = wav_tensor.shape[1]
        trim_at = samples
        for i in range(samples - 1, max(samples - max_trim, 0), -1):
            if abs(wav_tensor[0, i].item()) > threshold:
                trim_at = i + 1
                break
        return wav_tensor[:, :trim_at]

    pieces = []
    tail_silence = None
    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Turbo chunk %d/%d chars=%d text=%r", idx, len(chunks), len(chunk), chunk)
        part = generate_single_chunk(model, chunk, audio_prompt_path, exaggeration, cfg_weight, temperature)
        if idx < len(chunks):
            part = trim_trailing_silence(part, MODEL_SR)
            pause_samples = max(1, int(MODEL_SR * 0.120))
            pause = torch.zeros((1, pause_samples), dtype=part.dtype)
            pieces.append(part)
            pieces.append(pause)
        else:
            pieces.append(part)
        if tail_silence is None:
            tail_samples = max(1, int(MODEL_SR * 0.18))
            tail_silence = torch.zeros((1, tail_samples), dtype=part.dtype)

    if tail_silence is not None:
        pieces.append(tail_silence)

    return torch.cat(pieces, dim=1), chunks


def encode_audio_ffmpeg(wav_bytes: bytes, response_format: str, speed: float):
    response_format = normalize_response_format(response_format)
    if speed < 0.5 or speed > 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")

    suffix_map = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "pcm": ".pcm"}
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "pcm": "audio/pcm"}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        in_path = f_in.name
        f_in.write(wav_bytes)

    if response_format == "wav" and abs(speed - 1.0) < 1e-6:
        try:
            return wav_bytes, mime_map[response_format], suffix_map[response_format]
        finally:
            try:
                os.remove(in_path)
            except FileNotFoundError:
                pass

    out_path = in_path + suffix_map[response_format]

    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path]
        if abs(speed - 1.0) >= 1e-6:
            cmd += ["-filter:a", f"atempo={speed}"]

        if response_format == "wav":
            cmd += ["-c:a", "pcm_s16le", out_path]
        elif response_format == "mp3":
            cmd += ["-c:a", "libmp3lame", "-q:a", "2", out_path]
        elif response_format == "pcm":
            cmd += ["-f", "s16le", "-acodec", "pcm_s16le", out_path]
        else:
            cmd += ["-c:a", "flac", out_path]

        subprocess.run(cmd, check=True)
        with open(out_path, "rb") as f:
            return f.read(), mime_map[response_format], suffix_map[response_format]
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Streaming synthesis (2026-09-11): T3 speech tokens are generated one at a
# time; S3Gen renders a sliding token window as tokens arrive and only the new
# audio is emitted, overlap-added with the previous chunk's tail. First audio
# leaves after STREAM_FIRST_TOKENS tokens (~0.5 s of speech) instead of after
# the whole utterance. Non-streaming path is unchanged; TURBO_STREAM=0 or
# "stream": false in the request disables it.
# ---------------------------------------------------------------------------
STREAM_ENABLED = os.environ.get("TURBO_STREAM", "1").lower() not in ("0", "false", "no")
STREAM_FIRST_TOKENS = int(os.environ.get("TURBO_STREAM_FIRST_TOKENS", "20"))
STREAM_CHUNK_TOKENS = int(os.environ.get("TURBO_STREAM_CHUNK_TOKENS", "25"))
STREAM_CONTEXT_TOKENS = int(os.environ.get("TURBO_STREAM_CONTEXT_TOKENS", "25"))
STREAM_XFADE_MS = int(os.environ.get("TURBO_STREAM_XFADE_MS", "60"))
# S3Gen renders short windows a few dB hotter than a full utterance and every render
# varies by a few dB; the first chunk gets a fixed trim and later chunks are level-matched
# to the audio already emitted for their context region.
STREAM_FIRST_GAIN_DB = float(os.environ.get("TURBO_STREAM_FIRST_GAIN_DB", "-2.5"))
STREAM_MATCH_LIMIT_DB = float(os.environ.get("TURBO_STREAM_MATCH_LIMIT_DB", "6"))
# Hot outlier guard for the first chunk (full-utterance renders sit around -22..-31 dBFS RMS).
STREAM_FIRST_RMS_CEIL_DB = float(os.environ.get("TURBO_STREAM_FIRST_RMS_CEIL_DB", "-20"))
S3GEN_SIL = getattr(turbo_module, "S3GEN_SIL", 4299)
SPEECH_VOCAB = 6561
SAMPLES_PER_TOKEN = 960  # 24 kHz output / 25 Hz speech tokens
_GEN_LOCK = threading.Lock()
_CONDS_CACHE = {}


def _conds_for(model, audio_prompt_path: str):
    """prepare_conditionals is deterministic per reference file; cache it."""
    key = (audio_prompt_path, os.path.getmtime(audio_prompt_path))
    conds = _CONDS_CACHE.get(key)
    if conds is None:
        model.prepare_conditionals(audio_prompt_path, exaggeration=0.0)
        conds = model.conds
        _CONDS_CACHE[key] = conds
    model.conds = conds
    return conds


def _t3_token_stream(t3, t3_cond, text_tokens, temperature, top_k, top_p, repetition_penalty, max_gen_len=1000):
    """Mirror of T3.inference_turbo that yields each speech token as it is sampled."""
    logits_processors = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        logits_processors.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p))
    if repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )
    generated = []
    llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    past_key_values = llm_outputs.past_key_values
    speech_logits = t3.speech_head(llm_outputs[0][:, -1:])
    processed = logits_processors(speech_start_token, speech_logits[:, -1, :])
    next_tok = torch.multinomial(F.softmax(processed, dim=-1), num_samples=1)
    generated.append(next_tok)
    if torch.all(next_tok == t3.hp.stop_speech_token):
        return
    yield int(next_tok.item())
    current = next_tok
    for _ in range(max_gen_len):
        llm_outputs = t3.tfmr(inputs_embeds=t3.speech_emb(current), past_key_values=past_key_values, use_cache=True)
        past_key_values = llm_outputs.past_key_values
        speech_logits = t3.speech_head(llm_outputs[0])
        input_ids = torch.cat(generated, dim=1)
        processed = logits_processors(input_ids, speech_logits[:, -1, :])
        if torch.all(processed == -float("inf")):
            logger.warning("stream: all logits are -inf")
            break
        next_tok = torch.multinomial(F.softmax(processed, dim=-1), num_samples=1)
        generated.append(next_tok)
        current = next_tok
        if torch.all(next_tok == t3.hp.stop_speech_token):
            break
        yield int(next_tok.item())


@torch.inference_mode()
def stream_synthesize(model, text: str, audio_prompt_path: str, temperature: float):
    """Yield float32 numpy audio chunks (MODEL_SR) for one text chunk as tokens arrive."""
    _conds_for(model, audio_prompt_path)
    text = turbo_module.punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
    xf = max(1, int(MODEL_SR * STREAM_XFADE_MS / 1000))
    tokens = []
    state = {"emitted": 0, "tail": None, "hist": np.zeros(0, dtype=np.float32)}
    first_gain = 10.0 ** (STREAM_FIRST_GAIN_DB / 20.0)
    match_limit = 10.0 ** (STREAM_MATCH_LIMIT_DB / 20.0)

    def _rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0

    def synth(window_tokens):
        st = torch.tensor(window_tokens, dtype=torch.long, device=model.device)
        wav, _ = model.s3gen.inference(speech_tokens=st, ref_dict=model.conds.gen, n_cfm_timesteps=2)
        return wav.squeeze(0).float().cpu().numpy()

    def emit(end: int, final: bool):
        emitted = state["emitted"]
        start_ctx = max(0, emitted - STREAM_CONTEXT_TOKENS)
        ctx = emitted - start_ctx
        window = tokens[start_ctx:end] + ([S3GEN_SIL] * 3 if final else [])
        wav = synth(window)
        boundary = ctx * SAMPLES_PER_TOKEN
        if ctx > 0:
            old_ctx = state["hist"][start_ctx * SAMPLES_PER_TOKEN:emitted * SAMPLES_PER_TOKEN]
            r_old, r_new = _rms(old_ctx), _rms(wav[:boundary])
            gain = min(match_limit, max(1.0 / match_limit, r_old / r_new)) if r_old > 1e-4 and r_new > 1e-4 else 1.0
        else:
            gain = first_gain
            r0 = _rms(wav) * gain
            ceil = 10.0 ** (STREAM_FIRST_RMS_CEIL_DB / 20.0)
            if r0 > ceil:
                gain *= ceil / r0
        wav = wav * np.float32(gain)
        new = wav[boundary:]
        state["hist"] = np.concatenate([state["hist"], new[:(end - emitted) * SAMPLES_PER_TOKEN]])
        tail = state["tail"]
        if tail is not None and boundary >= len(tail):
            ovl = wav[boundary - len(tail):boundary]
            ramp = np.linspace(0.0, 1.0, len(tail), dtype=np.float32)
            out = np.concatenate([tail * (1.0 - ramp) + ovl * ramp, new])
        else:
            out = new
        if final or len(out) <= xf:
            state["tail"] = None
        else:
            state["tail"] = out[-xf:]
            out = out[:-xf]
        state["emitted"] = end
        return out

    for tok in _t3_token_stream(model.t3, model.conds.t3, text_tokens, temperature, 1000, 0.95, 1.2):
        if tok >= SPEECH_VOCAB:
            continue
        tokens.append(tok)
        n = len(tokens)
        if state["emitted"] == 0:
            if n >= STREAM_FIRST_TOKENS:
                yield emit(n, False)
        elif n - state["emitted"] >= STREAM_CHUNK_TOKENS:
            yield emit(n, False)
    if tokens and (len(tokens) > state["emitted"] or state["tail"] is not None):
        yield emit(len(tokens), True)


def _soft_limit(x: np.ndarray, knee: float = 0.84) -> np.ndarray:
    """Transparent below -1.5 dBFS; squashes peaks above it so short windows that
    render hot (S3Gen on a 20-token prefix runs a few dB louder than the full
    utterance) cannot hard-clip. Output stays within +/-1.0."""
    ax = np.abs(x)
    over = ax > knee
    if not over.any():
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (1.0 - knee) * np.tanh((ax[over] - knee) / (1.0 - knee)))
    return y


def _ffmpeg_stream_cmd(response_format: str, speed: float, target_sr: int):
    cmd = ["ffmpeg", "-loglevel", "error", "-nostdin", "-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0",
           "-f", "s16le", "-ar", str(MODEL_SR), "-ac", "1", "-i", "pipe:0"]
    if abs(speed - 1.0) >= 1e-6:
        cmd += ["-filter:a", f"atempo={speed}"]
    cmd += ["-ar", str(target_sr)]
    if response_format == "mp3":
        # No ID3/Xing headers: HA concatenates per-sentence responses into one stream and
        # its transcoder must not hit a second file header mid-stream.
        cmd += ["-c:a", "libmp3lame", "-q:a", "2", "-write_xing", "0", "-id3v2_version", "0", "-f", "mp3"]
    else:
        cmd += ["-f", "s16le", "-acodec", "pcm_s16le"]
    cmd += ["-flush_packets", "1", "pipe:1"]
    return cmd


def stream_response(model, text: str, audio_prompt_path: str, temperature: float, response_format: str, speed: float, target_sr: int):
    """Generator of encoded bytes: synthesis thread -> ffmpeg pipe -> HTTP chunks."""
    chunks = split_text_for_turbo(text)
    proc = subprocess.Popen(_ffmpeg_stream_cmd(response_format, speed, target_sr),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    t0 = time.time()
    stats = {"pieces": 0, "samples": 0, "first_audio": None, "error": None}

    def writer():
        try:
            with _GEN_LOCK:
                for idx, chunk in enumerate(chunks, start=1):
                    for piece in stream_synthesize(model, chunk, audio_prompt_path, temperature):
                        if stats["first_audio"] is None:
                            stats["first_audio"] = time.time() - t0
                        pcm = (_soft_limit(piece) * 32767.0).astype("<i2").tobytes()
                        proc.stdin.write(pcm)
                        proc.stdin.flush()
                        stats["pieces"] += 1
                        stats["samples"] += len(piece)
                    if idx < len(chunks):
                        proc.stdin.write(b"\x00" * (2 * int(MODEL_SR * CHUNK_PAUSE_MS / 1000)))
                proc.stdin.write(b"\x00" * (2 * int(MODEL_SR * 0.18)))
        except Exception as exc:  # noqa: BLE001
            stats["error"] = exc
            logger.exception("stream synthesis failed")
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    sent = 0
    first_byte = None
    try:
        while True:
            data = proc.stdout.read(2048)
            if not data:
                break
            if first_byte is None:
                first_byte = time.time() - t0
            sent += len(data)
            yield data
    finally:
        th.join(timeout=30)
        proc.wait(timeout=10)
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        logger.info(
            "Streamed %s bytes format=%s pieces=%d audio=%.2fs first_audio=%s first_byte=%s total=%.2fs%s",
            sent, response_format, stats["pieces"], stats["samples"] / float(MODEL_SR or 24000),
            f"{stats['first_audio']:.2f}s" if stats["first_audio"] is not None else "n/a",
            f"{first_byte:.2f}s" if first_byte is not None else "n/a",
            time.time() - t0, (" ffmpeg=" + err[:200]) if err else "",
        )
        if stats["error"] is not None:
            logger.error("stream ended with error: %s", stats["error"])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "default_voice": DEFAULT_VOICE,
        "chunk_threshold_chars": CHUNK_THRESHOLD_CHARS,
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "chunk_pause_ms": CHUNK_PAUSE_MS,
    })


@app.route("/v1/audio/speech", methods=["POST"])
def tts_speech():
    try:
        data = request.get_json(force=True, silent=False) or {}
        text = data.get("input") or data.get("text")
        if not text:
            return jsonify({"error": "Missing input/text field"}), 400

        voice = data.get("voice", DEFAULT_VOICE)
        audio_prompt_path = data.get("audio_prompt_path") or VOICE_MAP.get(voice, VOICE_MAP[DEFAULT_VOICE])
        exaggeration = as_float(data.get("exaggeration", 0.35), 0.35)
        cfg_weight = as_float(data.get("cfg_weight", 0.45), 0.45)
        temperature = as_float(data.get("temperature", 0.7), 0.7)
        speed = as_float(data.get("speed", 1.1), 1.1)
        response_format = normalize_response_format(data.get("response_format"))
        requested_sample_rate = int(data.get("sample_rate")) if data.get("sample_rate") else None
        normalized_text = normalize_whitespace(text)
        if fix_pronunciation is not None:
            normalized_text = fix_pronunciation(normalized_text)

        logger.info(
            "Request voice=%s format=%s speed=%.2f exaggeration=%.2f cfg=%.2f temp=%.2f chars=%d",
            voice, response_format, speed, exaggeration, cfg_weight, temperature, len(normalized_text)
        )

        model = load_model()
        want_stream = str(data.get("stream", "1")).lower() not in ("0", "false", "no")
        if STREAM_ENABLED and want_stream and response_format in ("pcm", "mp3"):
            target_sample_rate = resolve_target_sample_rate(response_format, requested_sample_rate)
            mime = "audio/mpeg" if response_format == "mp3" else "audio/pcm"
            return Response(
                stream_response(model, normalized_text, audio_prompt_path, temperature, response_format, speed, target_sample_rate),
                mimetype=mime,
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
                direct_passthrough=True,
            )
        t0 = time.time()
        wav, chunks = generate_with_chunking(
            model,
            normalized_text,
            audio_prompt_path,
            exaggeration,
            cfg_weight,
            temperature,
        )
        gen_s = time.time() - t0
        logger.info("Generated in %.2fs using %d chunk(s)", gen_s, len(chunks))

        target_sample_rate = resolve_target_sample_rate(response_format, requested_sample_rate)
        if MODEL_SR != target_sample_rate:
            wav = AF.resample(wav, MODEL_SR, target_sample_rate)

        buf = io.BytesIO()
        ta.save(buf, wav, target_sample_rate, format="wav")
        audio_bytes, mime_type, suffix = encode_audio_ffmpeg(buf.getvalue(), response_format, speed)

        return send_file(io.BytesIO(audio_bytes), mimetype=mime_type, as_attachment=False, download_name=f"speech{suffix}")
    except Exception as e:
        logger.exception("TTS request failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5051"))
    app.run(host="0.0.0.0", port=port)
