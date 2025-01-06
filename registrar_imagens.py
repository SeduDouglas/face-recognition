import chromadb
import rede_siamesa
import cv2
import chroma_facade
from pathlib import Path
import sys
import os
from image_utils import redimensionar_com_preenchimento, transform_to_tensor, recortar_redimensionar_com_preenchimento

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov9 import yolo_facade

chroma_facade.remover('id1')

weights=ROOT / 'model/yolov9-c-face.pt'

model = yolo_facade.load_model(weights)

model_siamesa = rede_siamesa.load_model()
imagem_base = cv2.imread('WIN_20241211_19_09_29_Pro.jpg')

face_results = yolo_facade.detect_from_image_data(model, imagem_base, conf_thres=0.40)

for *xyxy, conf, cls in face_results:
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    h, w = y2 - y1, x2 - x1
    frame_redimensionado = recortar_redimensionar_com_preenchimento(imagem_base, x1, y1, x2, y2, (128, 128))[0]
    tensor = transform_to_tensor(frame_redimensionado)
    embeding_frame_redimensionado = model_siamesa(tensor)
    chroma_facade.cadastrar(embeding_frame_redimensionado.detach().numpy(), ['id1'], ['Douglas'])