"""Pinned, CPU-only Silero speech trimming; no model download at runtime."""
from pathlib import Path
import threading
import numpy as np
from .audio import DecodedAudio


class SpeechTrimmer:
    def __init__(self, model_path: Path):
        import torch
        from silero_vad import get_speech_timestamps
        self.model = torch.jit.load(str(model_path), map_location='cpu').eval()
        self.timestamps = get_speech_timestamps
        self.lock = threading.Lock()  # Silero owns recurrent state across chunks.

    def trim(self, audio: DecodedAudio) -> DecodedAudio:
        import torch
        if audio.sample_rate not in {8000, 16000}:
            raise ValueError('speech trimming requires 8000 or 16000 Hz')
        with self.lock, torch.inference_mode():
            spans = self.timestamps(torch.from_numpy(audio.samples), self.model,
                                    sampling_rate=audio.sample_rate)
        samples = np.concatenate([audio.samples[s['start']:s['end']] for s in spans]) if spans else np.empty(0, dtype=np.float32)
        if samples.size < int(.8 * audio.sample_rate):
            raise ValueError('audio must contain at least 800 ms of speech')
        return DecodedAudio(np.ascontiguousarray(samples, dtype=np.float32), audio.sample_rate)
