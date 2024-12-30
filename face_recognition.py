import cv2
import cvzone
import sys
import os
import torch
from pathlib import Path
import torchvision.transforms as transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov9 import yolo_facade
from cv2_utils import recortar_imagem, redimensionar_com_preenchimento

video = r''
weights=ROOT / 'model/yolov9-c-face.pt'
weights_siamese=ROOT / 'model/trained_siamese_model.pt'

device = torch.device('cpu')

model = yolo_facade.load_model(weights)

model_siamesa = torch.load(weights_siamese, weights_only=False, map_location=device)
model_siamesa.eval()
imagem_base = cv2.imread('WIN_20241211_19_09_29_Pro.jpg')
imagem_base_redimensionada = redimensionar_com_preenchimento(imagem_base, (128, 128))[0]
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor
tensor = transform(imagem_base_redimensionada)
tensor = torch.unsqueeze(tensor, 0)
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
        frame_recortado = recortar_imagem(frame, x1, y1, w, h)
        frame_redimensionado = redimensionar_com_preenchimento(frame_recortado, (128, 128))[0]
        tensor = transform(frame_redimensionado)
        tensor = torch.unsqueeze(tensor, 0)
        embeding_frame_redimensionado = model_siamesa(tensor)
        distancia = ((embeding_imagem_base-embeding_frame_redimensionado)**2).sum(axis=1)
        print(distancia)
        cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()