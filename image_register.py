import chromadb
import siamese_network
import cv2
import chroma_facade
from pathlib import Path
import sys
import os
from image_utils import padding_resize, transform_to_tensor, crop_resize_padding

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


from yolov9 import yolo_facade

chroma_facade.remove('id1')

weights=ROOT / 'model/yolov9-c-face.pt'

model = yolo_facade.load_model(weights)

siamese_model = siamese_network.load_model()
base_image = cv2.imread('douglas.png')
base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)

face_results = yolo_facade.detect_from_image_data(model, base_image, conf_thres=0.40)

for *xyxy, conf, cls in face_results:
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    h, w = y2 - y1, x2 - x1
    resized_image = crop_resize_padding(base_image, x1, y1, x2, y2, (128, 128))[0]
    tensor = transform_to_tensor(resized_image)
    resized_frame_embedding = siamese_model(tensor)
    chroma_facade.register(resized_frame_embedding.detach().numpy(), ['id1'], ['Douglas'])