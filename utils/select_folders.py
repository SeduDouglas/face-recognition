import os
import shutil

def copiar_pastas_com_mais_de_um_arquivo(origem, destino):
    if not os.path.exists(destino):
        os.makedirs(destino)

    for item in os.listdir(origem):
        caminho_item = os.path.join(origem, item)

        if os.path.isdir(caminho_item):
            arquivos_na_pasta = [f for f in os.listdir(caminho_item) if os.path.isfile(os.path.join(caminho_item, f))]

            if len(arquivos_na_pasta) > 1:
                destino_pasta = os.path.join(destino, item)
                contador = 1
                while os.path.exists(destino_pasta):
                    destino_pasta = os.path.join(destino, f"{item}_{contador}")
                    contador += 1

                shutil.copytree(caminho_item, destino_pasta)
                print(f"Pasta {item} copiada para {destino_pasta}")

origem = "/caminho/para/pasta/origem"
destino = "/caminho/para/pasta/destino"

copiar_pastas_com_mais_de_um_arquivo(origem, destino)
