import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'vec-768-layer-9.onnx'
model_quant = 'vec-768-layer-9.quant.onnx'
quantized_model = quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QUInt8,
)
model_fp32 = 'my-model.onnx'
model_quant = 'my-model.quant.onnx'
quantized_model = quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QUInt8,
)
