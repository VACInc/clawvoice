"""Speech VADs for realtime PCM16 audio: Silero (default) and the legacy energy gate."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VADEvent:
    type: str
    audio_ms: int


class EnergyVAD:
    """Simple deterministic VAD over 16-bit PCM frames."""

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        threshold: float = 0.5,
        silence_duration_ms: int = 500,
        frame_ms: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = max(0.0, min(1.0, threshold))
        self.silence_duration_ms = max(0, silence_duration_ms)
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.rms_threshold = max(300.0, 150.0 + self.threshold * 1500.0)
        self.pending = bytearray()
        self.audio_ms = 0
        self.in_speech = False
        self.silence_ms = 0
        self.last_voice_end_ms = 0
        self.voiced_ms = 0

    def reset(self) -> None:
        self.pending.clear()
        self.audio_ms = 0
        self.in_speech = False
        self.silence_ms = 0
        self.last_voice_end_ms = 0
        self.voiced_ms = 0

    def reconfigure(
        self,
        *,
        sample_rate: int | None = None,
        threshold: float | None = None,
        silence_duration_ms: int | None = None,
    ) -> None:
        old_audio_ms = self.audio_ms
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if threshold is not None:
            self.threshold = max(0.0, min(1.0, threshold))
        if silence_duration_ms is not None:
            self.silence_duration_ms = max(0, silence_duration_ms)
        self.frame_bytes = int(self.sample_rate * self.frame_ms / 1000) * 2
        self.rms_threshold = max(300.0, 150.0 + self.threshold * 1500.0)
        self.audio_ms = old_audio_ms

    def feed(self, pcm16: bytes) -> list[VADEvent]:
        events: list[VADEvent] = []
        if not pcm16:
            return events
        self.pending.extend(pcm16)
        while len(self.pending) >= self.frame_bytes:
            frame = bytes(self.pending[: self.frame_bytes])
            del self.pending[: self.frame_bytes]
            rms = self._rms(frame)
            voiced = rms >= self.rms_threshold
            if voiced:
                self.voiced_ms += self.frame_ms
                if not self.in_speech:
                    self.in_speech = True
                    self.silence_ms = 0
                    events.append(VADEvent("speech_started", self.audio_ms))
                self.silence_ms = 0
                self.last_voice_end_ms = self.audio_ms + self.frame_ms
            elif self.in_speech:
                self.silence_ms += self.frame_ms
                if self.silence_ms >= self.silence_duration_ms:
                    self.in_speech = False
                    self.silence_ms = 0
                    events.append(VADEvent("speech_stopped", self.last_voice_end_ms))
            self.audio_ms += self.frame_ms
        return events

    @staticmethod
    def _rms(frame: bytes) -> float:
        if len(frame) < 2:
            return 0.0
        samples = np.frombuffer(frame, dtype="<i2")
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


@functools.lru_cache(maxsize=None)
def _silero_session(model_path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


class SileroVAD:
    """Silero v5 ONNX speech VAD with hysteresis and a minimum speech run.

    Breaths, sniffs, clicks, and steady background noise score low on Silero,
    and a single loud frame can no longer open a turn: speech_started fires only
    after ``min_speech_ms`` of frames at or above ``threshold``. Once in speech,
    frames down to ``threshold - 0.15`` still count as speech so word gaps do
    not end the turn early.
    """

    MODEL_SR = 16000
    WINDOW = 512  # 32 ms at 16 kHz
    CONTEXT = 64
    FRAME_MS = 32

    WARMUP_MS = 512
    LEVEL_FRAMES = 5  # ~160 ms loudness smoothing so single syllables do not flip the gate
    BG_DECAY_TAU_MS = 4000
    LAST_TURN_DECAY_TAU_MS = 10000
    ONSET_VS_LAST_TURN = 0.6

    def __init__(
        self,
        *,
        model_path: str | Path,
        sample_rate: int = 24000,
        threshold: float = 0.5,
        silence_duration_ms: int = 500,
        min_speech_ms: int = 200,
        loudness_ratio: float = 0.0,
        max_turn_ms: int = 15000,
        turn_drop_ratio: float = 0.0,
    ) -> None:
        self.session = _silero_session(str(model_path))
        self.sample_rate = sample_rate
        self.threshold = max(0.05, min(0.95, threshold))
        self.silence_duration_ms = max(0, silence_duration_ms)
        self.min_speech_ms = max(self.FRAME_MS, min_speech_ms)
        # Background speech (TV, other rooms) is real speech to Silero. Speech
        # only counts as the user when it is also ``loudness_ratio`` times louder
        # than the background speech level learned outside user turns (fades
        # when the background goes quiet). 0 disables the gate.
        self.loudness_ratio = max(0.0, loudness_ratio)
        self.max_turn_ms = max(0, max_turn_ms)
        # End of turn is relative to the speaker's own level: once the turn has
        # ~0.5 s of speech, frames quieter than ``turn_drop_ratio`` x the median of its last ~1.5 s
        # count as silence, so a quieter TV cannot hold the turn open. 0 disables.
        self.turn_drop_ratio = max(0.0, turn_drop_ratio)
        # The next turn must also open at >= ONSET_VS_LAST_TURN x the last
        # turn's level (fading over ~10 s), so the TV cannot start a turn, or
        # barge in on TARS, right after the user finishes. Off with turn_drop_ratio:
        # in real use (2026-09-27) a sentence trailing off quieter was cut mid-way.
        self.last_turn_rms = 0.0
        # Background level survives reset(): it describes the room, not the turn.
        self.ambient_rms = 0.0
        self.session_ms = 0
        self.energies: list[float] = []
        self.reset()

    @property
    def in_frame_samples(self) -> int:
        return int(self.sample_rate * self.FRAME_MS / 1000)

    def turn_median_rms(self) -> float:
        return float(np.median(self.turn_levels)) if self.turn_levels else 0.0

    def gate_rms(self) -> float:
        return self.ambient_rms * self.loudness_ratio

    def reset(self) -> None:
        self.pending = np.zeros(0, dtype=np.float32)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT, dtype=np.float32)
        self.audio_ms = 0
        self.in_speech = False
        self.silence_ms = 0
        self.last_voice_end_ms = 0
        self.voiced_ms = 0
        self.candidate_ms = 0
        self.candidate_start_ms: int | None = None
        self.turn_ms = 0
        self.turn_peak_rms = 0.0
        self.turn_levels: list[float] = []

    def reconfigure(
        self,
        *,
        sample_rate: int | None = None,
        threshold: float | None = None,  # noqa: ARG002 - client energy thresholds do not map to Silero probabilities
        silence_duration_ms: int | None = None,
    ) -> None:
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if silence_duration_ms is not None:
            self.silence_duration_ms = max(0, silence_duration_ms)

    def _to_model_rate(self, block: np.ndarray) -> np.ndarray:
        if self.sample_rate == self.MODEL_SR:
            return block
        positions = np.arange(self.WINDOW, dtype=np.float32) * (len(block) / self.WINDOW)
        return np.interp(positions, np.arange(len(block), dtype=np.float32), block).astype(np.float32)

    def _prob(self, window: np.ndarray) -> float:
        x = np.concatenate([self.context, window])[None, :]
        out, self.state = self.session.run(
            None,
            {"input": x, "state": self.state, "sr": np.array(self.MODEL_SR, dtype=np.int64)},
        )
        self.context = window[-self.CONTEXT :]
        return float(out[0][0])

    def feed(self, pcm16: bytes) -> list[VADEvent]:
        events: list[VADEvent] = []
        if not pcm16:
            return events
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        step = self.in_frame_samples
        neg_threshold = max(0.01, self.threshold - 0.15)
        decay = self.FRAME_MS / self.BG_DECAY_TAU_MS
        while len(self.pending) >= step:
            block, self.pending = self.pending[:step], self.pending[step:]
            prob = self._prob(self._to_model_rate(block))
            self.energies = (self.energies + [float(np.mean(block**2))])[-self.LEVEL_FRAMES :]
            rms = float(np.sqrt(np.mean(self.energies)))
            loud = (self.loudness_ratio == 0 or rms >= self.gate_rms()) and (self.turn_drop_ratio == 0 or rms >= self.last_turn_rms * self.ONSET_VS_LAST_TURN)
            warming = self.loudness_ratio > 0 and self.session_ms < self.WARMUP_MS
            frame_start = self.audio_ms
            if not self.in_speech:
                if warming:
                    # Learn the room before the first turn so an already-running TV cannot open it.
                    if prob >= neg_threshold:
                        self.ambient_rms = max(self.ambient_rms, rms)
                elif prob >= self.threshold and loud:
                    if self.candidate_start_ms is None:
                        self.candidate_start_ms = frame_start
                        self.turn_peak_rms = 0.0
                        self.turn_levels = []
                    self.candidate_ms += self.FRAME_MS
                    self.turn_peak_rms = max(self.turn_peak_rms, rms)
                    self.turn_levels.append(rms)
                    if self.candidate_ms >= self.min_speech_ms:
                        self.in_speech = True
                        self.silence_ms = 0
                        self.turn_ms = self.candidate_ms
                        self.voiced_ms += self.candidate_ms
                        self.last_voice_end_ms = frame_start + self.FRAME_MS
                        events.append(VADEvent("speech_started", self.candidate_start_ms))
                else:
                    if prob >= neg_threshold:  # background speech, too quiet to be the user: track its peaks
                        self.ambient_rms = max(self.ambient_rms * (1 - decay), rms)
                    else:
                        self.ambient_rms *= 1 - decay
                    if prob < neg_threshold or not loud:
                        self.candidate_ms = 0
                        self.candidate_start_ms = None
            else:
                self.turn_ms += self.FRAME_MS
                # Word tails are quieter than onsets, so the in-turn gate is 60% of the onset gate.
                own_floor = float(np.median(self.turn_levels[-47:])) * self.turn_drop_ratio if len(self.turn_levels) >= 16 else 0.0
                if prob >= neg_threshold and rms >= own_floor and (self.loudness_ratio == 0 or rms >= self.gate_rms() * 0.6):
                    self.silence_ms = 0
                    self.last_voice_end_ms = frame_start + self.FRAME_MS
                    self.turn_peak_rms = max(self.turn_peak_rms, rms)
                    if prob >= self.threshold:
                        self.voiced_ms += self.FRAME_MS
                        self.turn_levels.append(rms)
                else:
                    self.silence_ms += self.FRAME_MS
                if self.silence_ms >= self.silence_duration_ms or (self.max_turn_ms and self.turn_ms >= self.max_turn_ms):
                    end_ms = self.last_voice_end_ms if self.silence_ms else frame_start + self.FRAME_MS
                    self.last_turn_rms = self.turn_median_rms()
                    self.in_speech = False
                    self.silence_ms = 0
                    self.candidate_ms = 0
                    self.candidate_start_ms = None
                    events.append(VADEvent("speech_stopped", end_ms))
            if not self.in_speech:
                self.last_turn_rms *= 1 - self.FRAME_MS / self.LAST_TURN_DECAY_TAU_MS
            self.audio_ms += self.FRAME_MS
            self.session_ms += self.FRAME_MS
        return events
