import time
import numpy as np
import cv2
from tritonclient.utils import *
import tritonclient.grpc as grpcclient


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def Inference(IMAGE_PATH):
    SERVER_URL = 'BoonMoSa_TritonInferenceServer:8001'
    MODEL_NAME = 'BoonMoSa'

    dectection_image_path = 'outputs/' + IMAGE_PATH.split('.')[-2] + "-seg.png"
    dectection_boxes_path = 'outputs/' + IMAGE_PATH.split('.')[-2] + "-seg.txt"
    IMAGE_PATH = 'inputs/' + IMAGE_PATH

    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_image, r, _ = letterbox(image)
    input_image = input_image.astype('float32')
    input_image = input_image.transpose((2,0,1))[np.newaxis, :] / 255.0
    input_image = np.ascontiguousarray(input_image)
    with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
        inputs = [
            grpcclient.InferInput("images__0", input_image.shape, np_to_triton_dtype(np.float32))
        ]

        inputs[0].set_data_from_numpy(input_image)

        outputs = [
            grpcclient.InferRequestedOutput("output__0"),
            grpcclient.InferRequestedOutput("output__1")
        ]

        response = triton_client.infer(
                                    model_name=MODEL_NAME,
                                    inputs=inputs,
                                    outputs=outputs
                                    )

        response.get_response()
        output0 = response.as_numpy("output__0")
        output1 = response.as_numpy("output__1")
    return image, r, output0, output1, dectection_image_path, dectection_boxes_path

def main(IMAGE_PATH):
    START = time.time()
    image, r, output0, output1, dectection_image_path, dectection_boxes_path = Inference(IMAGE_PATH)
    results = output0.copy()
    protos = output1.copy()
    overlay = image.copy()

    ll = 1

    results[0, :, 0] = (results[0, :, 0] - results[0, :, 2] / 2) / r[0]
    results[0, :, 1] = (results[0, :, 1] - results[0, :, 3] / 2) / r[1]
    results[0, :, 2] /= r[0]
    results[0, :, 3] /= r[1]

    bboxes = results[0, :, :4]
    confidences = results[0, :, 4]
    scores = confidences.reshape(-1, 1) * results[0, :, 5:ll+5]
    masks = results[0, :, ll+5:]

    CONF_THRESHOLD = 0.03
    IOU_THRESHOLD = 0.5
    MASK_THRESHOLD = 17.5

    indices = cv2.dnn.NMSBoxes(
        bboxes.tolist(), confidences.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)

    color=(255, 0, 0)
    thickness=2

    for i in indices:
        bbox = bboxes[i].round().astype(np.int32)
        _, score, _, class_id = cv2.minMaxLoc(scores[i])
        class_id = class_id[1]
        if score >= CONF_THRESHOLD:
            c = bbox[:2]
            h = bbox[2:]
            p1, p2 = c, (c + h)
            p1, p2 = p1.astype('int32'), p2.astype('int32')
            cv2.rectangle(image, p1, p2, color, thickness)
            x,y,w,h = map(int, bbox * np.array([r[0], r[1], r[0], r[1]]) * 160 / 640)
            proto = protos[0,:,y:y+h,x:x+w].reshape(32, -1)
            np.expand_dims(masks[i], 0) @ proto
            proto = (1 / (1 + np.exp(-proto))).sum(0)
            proto = proto.reshape(h, w)
            mask = cv2.resize(proto, (bbox[2], bbox[3]))
            mask = mask >= MASK_THRESHOLD

            to_mask = overlay[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            mask = mask[:to_mask.shape[0], :to_mask.shape[1]]
            to_mask[mask] = [255, 0, 0]
    else:
        alpha = 0.5
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        cv2.imwrite(dectection_image_path, image[:, :, ::-1])
    END = time.time()
    return END - START


if __name__ == "__main__":
    main("test.png")