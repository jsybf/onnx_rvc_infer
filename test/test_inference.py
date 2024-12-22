# 로컬에서 rvc 음성합성 테스트 할 수 있는 코드
import sys

sys.path.append("..")

import src.rvc_onnx_infer as infer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

conf = infer.RvcConfig(vec_path=Path.cwd().parent / 'assets' / 'vec-768-layer-9.quant.onnx',
                       model_path=Path.cwd().parent / 'assets' / 'my-model.onnx',
                       device="cpu",
                       sampling_rate=48000,
                       hop_size=128,
                       f0_method="dio",
                       f0_up_key=0,
                       log_level=2
                       )

input_wav = open(Path.cwd().parent / 'sample-data' / 'Conference.wav', mode='rb').read()

output_wav = infer.convert_voice(input_wav, conf)

with open(Path.cwd() / 'output.wav', mode='wb') as f:
    f.write(output_wav)
