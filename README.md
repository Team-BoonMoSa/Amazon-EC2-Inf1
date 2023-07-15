![Amazon_EC2_Inf1](https://github.com/Zerohertz/zerohertz.github.io/assets/42334717/d1cbc5e5-e0a1-4763-adeb-6657568a6a85)

![END](https://github.com/Zerohertz/zerohertz.github.io/assets/42334717/62db6a0b-ce2d-4e4b-92a9-75598d0de5b3)

---

# Setup

```shell
.
├── README.md
├── client
│   ├── Dockerfile
│   └── app
│       ├── app.py
│       ├── client.py
│       └── requirements.txt
├── docker-compose.yaml
├── server
│   ├── Dockerfile
│   └── model-repository
│       ├── BoonMoSa
│       │   ├── 1
│       │   │   ├── model.py
│       │   │   └── model_neuron.pt.keep -> model_neuron.py로 교체!
│       │   └── config.pbtxt
│       └── gen_triton_model.py
└── sh
    ├── curl.sh
    ├── gen.sh
    └── rm.sh
```

```shell
$ docker compose up -d
```
