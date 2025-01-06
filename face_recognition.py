import cv2
import cvzone
import sys
import os
import torch
from pathlib import Path
import rede_siamesa
import chroma_facade

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov9 import yolo_facade
from image_utils import redimensionar_com_preenchimento, recortar_redimensionar_com_preenchimento, transform_to_tensor

weights=ROOT / 'model/yolov9-c-face.pt'

model = yolo_facade.load_model(weights)
model_siamesa = rede_siamesa.load_model()
imagem_base = cv2.imread('WIN_20241211_19_09_29_Pro.jpg')
imagem_base_redimensionada = redimensionar_com_preenchimento(imagem_base, (128, 128))[0]


# Convert the image to PyTorch tensor
tensor = transform_to_tensor(imagem_base_redimensionada)
embeding_imagem_base = model_siamesa(tensor)
print(embeding_imagem_base)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    face_results = yolo_facade.detect_from_image_data(model, frame, conf_thres=0.40)

    for *xyxy, conf, cls in face_results:
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        h, w = y2 - y1, x2 - x1
        frame_redimensionado = recortar_redimensionar_com_preenchimento(frame, x1, y1, x2, y2, (128, 128))[0]
        tensor = transform_to_tensor(frame_redimensionado)
        embeding_frame_redimensionado = model_siamesa(tensor)
        resultado = chroma_facade.buscar_proximos(embeding_frame_redimensionado.detach().numpy(), 10)
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