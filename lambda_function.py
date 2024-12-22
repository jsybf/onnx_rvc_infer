import base64
import io
import logging
import os
from pathlib import Path
from timeit import default_timer as timer

import soundfile

from src import rvc_onnx_infer as infer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(funcName)-25s %(message)s')
logger = logging.getLogger(__name__)


def parse_event(event: dict) -> tuple:
    try:
        wav_file = base64.b64decode(event['wav_file'])
    except Exception as e:
        logger.error(f"exeption during parsing event exception_message:{str(e)}")
        raise e

    return wav_file, None


hubert_path: Path = Path.cwd() / 'assets' / 'vec-768-layer-9.onnx'
g_net_path: Path = Path.cwd() / 'assets' / 'my-model.onnx'

# TODO: assert 가 최선인가? NUMBA_CACHE_DIR 가 none이면 종료
assert hubert_path.exists()
assert g_net_path.exists()

# RVC model config
conf = infer.RvcConfig(vec_path=hubert_path,
                       model_path=g_net_path,
                       device="cpu",
                       sampling_rate=48000,
                       hop_size=int(os.getenv("RVC_HOP_SIZE", 128)),
                       f0_method="dio",
                       f0_up_key=0,
                       log_level=int(os.getenv("RVC_ONNX_LOG_LEVEL", 2))
                       )


def handler(event: dict, context):
    # logging info and check environments
    logger.info(
        f"""
        
    "*************** env *************** "
    NUMBA_CACHE_DIR(required): {os.getenv('NUMBA_CACHE_DIR', None)}
    lambda memory allocated(MB): {context.memory_limit_in_mb}
    lambda remaining time(ms): {context.get_remaining_time_in_millis()}
    event.keys: {event.keys()} 
    rvc_configuration: {str(conf)} 
    
    
    """
    )

    # parse event object
    wav_file, _ = parse_event(event)
    assert wav_file is not None

    model = infer.OnnxRVC(conf.model_path, vec_path=conf.vec_path, sr=conf.sampling_rate, hop_size=conf.hop_size,
                          device=conf.device, log_level=conf.log_level)

    #  converse voice
    infer_start_time = timer()
    audio = model.inference(io.BytesIO(wav_file), 0, f0_method=conf.f0_method, f0_up_key=conf.f0_up_key)
    infer_end_time = timer()
    logger.info(f"inference duration: {infer_end_time - infer_start_time}")

    # return wav
    output_wav = io.BytesIO()
    soundfile.write(output_wav, audio, conf.sampling_rate, format='WAV')

    # encode wav file before sending
    output_wav_b64 = base64.b64encode(output_wav.getvalue()).decode('utf-8')

    return {
        "wav_b64": output_wav_b64
    }
