
from pathlib import Path
import sys
import torch
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

def load_model(weights_siamese = ROOT / 'model/trained_siamese_model.pt', device = 'cpu'):
    model_siamesa = torch.load(weights_siamese, weights_only=False, map_location=device)
    model_siamesa.eval()
    return model_siamesa