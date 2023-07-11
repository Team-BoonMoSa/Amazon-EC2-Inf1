from fastapi import FastAPI

from client import main

app = FastAPI()

@app.get("/inference")
def inference(file_id: str='test.png'):
    pt, bboxes, masks, labels = main(file_id)
    return {
        "request_info": {
            "file_id": file_id,
            "process_time": pt
        },
        "results": {
            "bboxes": [],
            "masks": [],
            "labels": labels
        }
    }
