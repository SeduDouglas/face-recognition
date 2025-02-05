import chromadb
import siamese_network
import cv2
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

def search_nearest(embeddings, result_count = 1):
    client = chromadb.PersistentClient(path="/db")

    if client.count_collections() == 0:
        collection = client.create_collection(name="faces_collection")

    collection = client.get_collection(name="faces_collection")

    result = collection.query(
        query_embeddings=embeddings,
        n_results=result_count
    )
    return result

def register(embeddings, ids, names):
    client = chromadb.PersistentClient(path="/db")

    if client.count_collections() == 0:
        collection = client.create_collection(name="faces_collection")

    collection = client.get_collection(name="faces_collection")

    collection.add(
        embeddings=embeddings,
        metadatas= [{"name": name} for name in names],
        ids=ids, 
    )

def remove(ids):
    client = chromadb.PersistentClient(path="/db")

    if client.count_collections() == 0:
        collection = client.create_collection(name="faces_collection")

    collection = client.get_collection(name="faces_collection")

    collection.delete(
        ids=ids, 
    )

