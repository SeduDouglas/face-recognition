import chromadb
import siamese_network
import cv2
import chroma_facade
from pathlib import Path
import sys
import os
from image_utils import padding_resize, transform_to_tensor, crop_resize_padding, FaceTensorTransform

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

origin = 'data/faces/register'
transform = FaceTensorTransform(128)
siamese_model = siamese_network.load_model()

chroma_facade.clean_collection()
count_dir = 0
for item in os.listdir(origin):
    item_path = os.path.join(origin, item)
    count_img = 0
    count_dir += 1
    if os.path.isdir(item_path):
        folder_files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
        for file in folder_files:
            count_img += 1
            base_image = cv2.imread(os.path.join(item_path, file))
            cropped_image, face_found = transform(base_image)
            if face_found:
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                tensor = transform_to_tensor(cropped_image)
                embedding = siamese_model(tensor)
                chroma_facade.register(embedding.detach().numpy(), [f'id{count_dir}{count_img}'], [item])
                