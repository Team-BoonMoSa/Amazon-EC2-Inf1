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

---

# Build Triton Inference Server

```shell
$ sh build_server.sh
```

```shell
$ sh rm_server.sh
```
