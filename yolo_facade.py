import cv2
import cvzone
import sys
import os
from pathlib import Path
import argparse
import os
import platform
import sys
import numpy as np
import torch
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams


def load_model(weights, dnn=False, data=ROOT / 'data/coco.yaml', fp16=False):
  device = select_device('cpu')
  model = DetectMultiBackend(weights, dnn=False, device=device, data=ROOT / 'data/coco.yaml', fp16=False)
  img_size = check_img_size((640, 640), s=model.stride)
  model.warmup(imgsz=(1 if model.pt or model.triton else 1, 3, *img_size))  # warmup

  return model

#Método criado a partir do método utilizado pelo Yolo, detect_dual.
@smart_inference_mode()
def detect_from_image_data(
  model,
  img_data,
  imgsz=(640, 640),  # inference size (height, width)
  conf_thres=0.25,  # confidence threshold
  iou_thres=0.45,  # NMS IOU threshold
  max_det=1000,  # maximum detections per image
  classes=None,  # filter by class: --class 0, or --class 0 2 3
  agnostic_nms=False,  # class-agnostic NMS
):

  stride, names, pt = model.stride, model.names, model.pt
  imgsz = check_img_size(imgsz, s=stride)  # check image size

  #view_img = check_imshow(warn=True)

  # Run inference
  dt = (Profile(), Profile(), Profile())

  im = transform_image(img_data, imgsz, stride, pt)

  with dt[0]:
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

  # Inference
  with dt[1]:
    pred = model(im, augment=False, visualize=False)
    pred = pred[0][1]

  # NMS
  with dt[2]:
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

  # Second-stage classifier (optional)
  # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

  # Process predictions
  det = pred[0]
  gn = torch.tensor(img_data.shape)[[1, 0, 1, 0]]  # normalization gain whwh
  if len(det):
    # Rescale boxes from img_size to image_data size
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img_data.shape).round()

  return reversed(det)


def transform_image(im0, img_size, stride, auto):
  im = letterbox(im0, img_size, stride, auto)[0]  # padded resize
  im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
  im = np.ascontiguousarray(im)  # contiguous

  return im