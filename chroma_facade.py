import chromadb
import rede_siamesa
import cv2
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

def buscar_proximos(embeddings, quantidade_resultados = 1):
    client = chromadb.PersistentClient(path="/db")

    if client.count_collections() == 0:
        collection = client.create_collection(name="faces_collection")

    collection = client.get_collection(name="faces_collection")

    result = collection.query(
        query_embeddings=embeddings,
        n_results=quantidade_resultados
    )
    return result

def cadastrar(embeddings, ids, nomes):
    client = chromadb.PersistentClient(path="/db")

    if client.count_collections() == 0:
        collection = client.create_collection(name="faces_collection")

    collection = client.get_collection(name="faces_collection")

    collection.add(
        embeddings=embeddings,
        metadatas= [{"nome": nome} for nome in nomes],
        ids=ids, 
    )

def remover(ids):
    client = chromadb.PersistentClient(path="/db")

    if client.count_collections() == 0:
        collection = client.create_collection(name="faces_collection")

    collection = client.get_collection(name="faces_collection")

    collection.delete(
        ids=ids, 
    )

