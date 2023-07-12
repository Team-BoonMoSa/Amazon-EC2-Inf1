![local](https://github.com/Team-BoonMoSa/.github/assets/42334717/74184fdc-5d7d-4daf-9a39-e079b93af1b3)

---

# Setup

```
.
├── ...
├── server
│   ├── Dockerfile
│   └── model-repository
│       └── BoonMoSa
│           ├── 1
│           │   └── model.onnx -> 학습된 모델로 교체!
│           └── config.pbtxt
├── client
│   ├── Dockerfile
│   ├── app
│   │   ├── app.py
│   │   ├── client.py
│   │   └── requirements.txt
│   ├── inputs
│   │   └── test.png
│   └── outputs
│       └── test-seg.png
└── docker-compose.yaml
```

# Build Triton Inference Server

```shell
$ docker compose up -d
```