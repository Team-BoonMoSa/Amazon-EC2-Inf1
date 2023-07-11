docker build -t boonmosa_client:dev ./client
docker run \
    -itd \
    --name BoonMoSa_FastAPI \
    -p 80:80 \
    -v ${PWD}/client/inputs:/app/inputs \
    -v ${PWD}/client/outputs:/app/outputs \
    boonmosa_client:dev
docker logs -f BoonMoSa_FastAPI