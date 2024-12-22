docker build --provenance=false --platform linux/amd64 -t rvc_onnx_infer_lambda:latest .
 docker run --rm -p 8080:8080 -t -e RVC_HOP_SIZE=64 rvc_onnx_infer_lambda:latest 