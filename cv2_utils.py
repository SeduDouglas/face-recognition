import cv2
import numpy as np

def redimensionar_com_preenchimento(im, nova_forma=(640, 640), color=(114, 114, 114), scaleup=True):
    # Redimensiona imagem e preenche os espaços
    shape = im.shape[:2]
    if isinstance(nova_forma, int):
        nova_forma = (nova_forma, nova_forma)

    r = min(nova_forma[0] / shape[0], nova_forma[1] / shape[1])
    if not scaleup: #caso não seja para aumentar a imagem
        r = min(r, 1.0)

    # Calcula o preenchimento
    ratio = r, r  # razão largura altura
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = nova_forma[1] - new_unpad[0], nova_forma[0] - new_unpad[1]  # preenchimento em altura e largura

    dw /= 2  # divide preenchimento em 2
    dh /= 2

    if shape[::-1] != new_unpad:  # redimensiona
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # adiciona borda
    return im, ratio, (dw, dh)

def recortar_imagem(img, start_x, start_y, width, height):
    # recorta imagem
    img_recortada = img[start_y:start_y+height, start_x:start_x+width]
    
    return img_recortada