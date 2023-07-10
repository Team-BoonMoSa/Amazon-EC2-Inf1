docker build -t boonmosa:dev .
docker run -itd \
--pid=host \
--shm-size=4gb \
-p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v ${PWD}/model-repository:/model-repository \
--name BoonMoSa_TritonInferenceServer \
boonmosa:dev \
tritonserver --model-repository=/model-repository --strict-model-config=false --log-verbose=1 --backend-config=python,grpc-timeout-milliseconds=50000 && \
docker logs -f BoonMoSa_TritonInferenceServer