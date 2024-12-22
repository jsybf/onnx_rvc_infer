import dataclasses
from timeit import default_timer as timer

import io
from pathlib import Path

import soundfile

from src.modules.onnx_inference import OnnxRVC
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(funcName)-25s %(message)s', force=True)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class RvcConfig:
    vec_path: Path  # pretrained ONNX variant of vec
    model_path: Path  # Your .ONNX model
    # wav_path: str
    device: str  # options: dml, cuda, cpu
    sampling_rate: int  # Your model's sample rate;   32000, 40000, 48000
    hop_size: int  # hop size for inference. ( Currently, applies only to dio F0 ) Try: 32, 64, 128, 256, 512  or custom of your choice
    f0_method: str  # F0 pitch estimation method.  ( For now only dio works properly. PM is fixed and works, but Dio is better. Harvest is broken )
    f0_up_key: int  # transpose; in semitones either up or down.
    log_level: int


def convert_voice(input_wav: bytes, conf: RvcConfig) -> bytes:
    # init onnx rvc model
    model = OnnxRVC(conf.model_path, vec_path=conf.vec_path, sr=conf.sampling_rate, hop_size=conf.hop_size,
                    device=conf.device, log_level=conf.log_level)

    #  converse voice
    infer_start_time= timer()
    audio = model.inference(io.BytesIO(input_wav), 0, f0_method=conf.f0_method, f0_up_key=conf.f0_up_key)
    infer_end_time = timer()
    logger.info(f"inference duration: {infer_end_time - infer_start_time}")

    # return wav
    output_wav = io.BytesIO()
    soundfile.write(output_wav, audio, conf.sampling_rate, format='WAV')
    return output_wav.getvalue()
