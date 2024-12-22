FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ${LAMBDA_TASK_ROOT}

# librosa use numba to cache data. However numba  tries to write cache file
# under ${LAMBDA_TASK_ROOT} which is read only file. set NUMBA_CACHE_DIR to prevent this.
ENV NUMBA_CACHE_DIR="/tmp"

RUN yum update -y &&\
    yum install -y \
    gcc-c++ \
    libsndfile

RUN  pip install --no-cache-dir --upgrade pip wheel && \
     pip install --no-cache-dir -r requirements.txt


COPY ./assets/vec-768-layer-9.onnx ${LAMBDA_TASK_ROOT}/assets/vec-768-layer-9.onnx
COPY ./assets/my-model.onnx ${LAMBDA_TASK_ROOT}/assets/my-model.onnx
COPY ./src ${LAMBDA_TASK_ROOT}/src
COPY ./lambda_function.py ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.handler" ]
