![local](https://github.com/Team-BoonMoSa/.github/assets/42334717/74184fdc-5d7d-4daf-9a39-e079b93af1b3)

---

# Setup

```shell
$ conda create -n triton python=3.8 -y
$ pip install -r requirements.txt
```

```
.
├── ...
└── model-repository
    └── BoonMoSa
        ├── 1
        │   └── model.onnx -> 학습된 모델로 교체!
        └── config.pbtxt
```

# Build Triton Inference Server

```shell
$ docker compose up -d
```