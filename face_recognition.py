import cv2
import cvzone
import sys
import os
import torch
from pathlib import Path
import siamese_network
import chroma_facade

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov9 import yolo_facade
from image_utils import padding_resize, crop_resize_padding, transform_to_tensor

weights=ROOT / 'model/yolov9-c-face.pt'

model = yolo_facade.load_model(weights)
siamese_model = siamese_network.load_model()

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    face_results = yolo_facade.detect_from_image_data(model, frame, conf_thres=0.40)

    for *xyxy, conf, cls in face_results:
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        h, w = y2 - y1, x2 - x1
        resized_frame = crop_resize_padding(frame, x1, y1, x2, y2, (128, 128))[0]
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        tensor = transform_to_tensor(resized_frame)
        resized_frame_embedding = siamese_model(tensor)
        resultado = chroma_facade.buscar_proximos(resized_frame_embedding.detach().numpy(), 10)
        print(resultado)
        cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
        cv2.putText(frame, resultado['metadatas'][0][0]['nome'],(x1,y2 - 5), cv2.FONT_HERSHEY_DUPLEX, 1,(255,0,255),2,cv2.LINE_AA)

    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()