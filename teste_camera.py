import cv2
import cvzone
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov9 import ModelUtil
import detect_from_image_data as detect_from_image_data
from image_utils import recortar_imagem

video = r''
weights=ROOT / 'model/yolov9-c-face.pt'

model = ModelUtil.load(weights)

# modelPath = 'models\yolov8n-face.pt'
# modelPath = 'models\yolov9-c-face.pt'

# facemodel = YOLO(modelPath)
# feceModelEmbedding = YOLO(modelPath)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    face_results = detect_from_image_data.run(model, frame, conf_thres=0.40)

    for *xyxy, conf, cls in face_results:
      x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
      h, w = y2 - y1, x2 - x1
      frame = recortar_imagem(frame, x1, y1, w, h);
      cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()