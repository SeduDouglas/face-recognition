import os
import shutil
import cv2
import image_utils
from PIL import Image

transform = image_utils.FaceTensorTransform(128)

def convert_to_training(origin, detiny):
    if not os.path.exists(detiny):
        os.makedirs(detiny)

    for item in os.listdir(origin):
        item_path = os.path.join(origin, item)

        if os.path.isdir(item_path):
            folder_files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
            destiny_folder = os.path.join(destino, item)
            if not os.path.exists(destiny_folder):
                os.makedirs(destiny_folder)
            for file in folder_files:
                base_image = cv2.imread(os.path.join(item_path, file))
                cropped_image, encontrou_face = transform(base_image)
                if encontrou_face:
                    abs_path = os.path.abspath(destiny_folder)
                    cv2.imwrite(os.path.join(abs_path, file), cropped_image)

origem = "data/faces/training"
destino = "data/faces/tratado"

convert_to_training(origem, destino)
