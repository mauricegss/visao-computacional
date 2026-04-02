import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

input_folder = "assets"
output_folder = "resultados_rgb"

os.makedirs(output_folder, exist_ok=True)

extensoes = (".jpg", ".webp")

def aplica_canny_rgb(img_suavizada):
    # Aplica Canny em cada canal isoladamente e une os resultados
    b, g, r = cv2.split(img_suavizada)
    canny_b = cv2.Canny(b, 100, 200)
    canny_g = cv2.Canny(g, 100, 200)
    canny_r = cv2.Canny(r, 100, 200)
    bordas_unidas = cv2.bitwise_or(canny_b, canny_g)
    bordas_unidas = cv2.bitwise_or(bordas_unidas, canny_r)
    return bordas_unidas

for nome in os.listdir(input_folder):
    if nome.lower().endswith(extensoes):
        caminho = os.path.join(input_folder, nome)
        
        img = cv2.imread(caminho)
        if img is None:
            continue
            
        # Converte BGR (OpenCV) para RGB (Matplotlib)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Filtro da Mediana em RGB
        img_suavizada = cv2.medianBlur(img_rgb, 5)
        
        # Detecção nas imagens coloridas
        bordas_canny_rgb = aplica_canny_rgb(img_suavizada)
        
        # Plot
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original (RGB)")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(bordas_canny_rgb, cmap='gray')
        plt.title("Detector Canny (Merge Canais RGB)")
        plt.axis('off')
        
        plt.tight_layout()
        
        saida = os.path.join(output_folder, f"resultado_rgb_{nome}.png")
        plt.savefig(saida)
        plt.close()
        
        print(f"Processado (RGB): {nome}")

print("✅ Tudo pronto no RGB!")