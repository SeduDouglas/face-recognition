import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
from pathlib import Path
import sys
import os


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from yolov9 import yolo_facade

def padding_resize(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2  
    dh /= 2

    if shape[::-1] != new_unpad:  
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def crop(img, start_x, start_y, end_x, end_y):

    img_recortada = img[start_y:end_y, start_x:end_x]
    
    return img_recortada

def crop_resize_padding(img, start_x, start_y, width, height, nova_forma=(640, 640), color=(114, 114, 114), scaleup=True):
    return padding_resize(crop(img, start_x, start_y, width, height), nova_forma, color, scaleup)

def transform_to_tensor(img):
    transform = transforms.ToTensor()
    tensor = transform(img)
    tensor = torch.unsqueeze(tensor, 0)
    return tensor


class FaceTensorTransform(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        weights=ROOT / 'model/yolov9-c-face.pt'
        self.output_size = output_size
        self.yolo_model = yolo_facade.load_model(weights)

    def __call__(self, sample):
        imagem_recortada = {}
        encontrou_face = False
        sample_np = np.array(sample)
        melhor_face = {}
        melhor_confiabilidade = 0
        face_results = yolo_facade.detect_from_image_data(self.yolo_model, sample_np, conf_thres=0.40)
        for face in face_results:
            encontrou_face = True
            *xyxy, conf, cls = face
            if melhor_confiabilidade == 0 or melhor_confiabilidade < conf:
                melhor_face = face
        if encontrou_face:
            *xyxy, conf, cls = melhor_face
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            imagem_recortada = crop_resize_padding(sample_np, x1, y1, x2, y2, (128, 128))[0]

        return imagem_recortada, encontrou_face
