import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

input_folder = "assets-grayscale"
output_folder = "resultados"

os.makedirs(output_folder, exist_ok=True)

for nome in os.listdir(input_folder):
    caminho = os.path.join(input_folder, nome)
    
    img = cv2.imread(caminho, 0)
    if img is None:
        continue
    
    # 1. Pré-processamento: Filtro da Mediana (preserva as bordas melhor que a média)
    img_suavizada = cv2.medianBlur(img, 5)
    
    # 2. Detector Canny (Obrigatório)
    bordas_canny = cv2.Canny(img_suavizada, 100, 200)
    
    # 3. Detector Sobel (Escolha 1)
    sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)
    bordas_sobel = cv2.magnitude(sobel_x, sobel_y)
    bordas_sobel = cv2.convertScaleAbs(bordas_sobel)
    
    # 4. Detector Laplaciano (Escolha 2)
    bordas_laplaciano = cv2.Laplacian(img_suavizada, cv2.CV_64F)
    bordas_laplaciano = cv2.convertScaleAbs(bordas_laplaciano)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original (Cinza)")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(bordas_canny, cmap='gray')
    plt.title("Detector Canny")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(bordas_sobel, cmap='gray')
    plt.title("Detector Sobel")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(bordas_laplaciano, cmap='gray')
    plt.title("Detector Laplaciano")
    plt.axis('off')
    
    plt.tight_layout()
    
    saida = os.path.join(output_folder, f"resultado_{nome}.png")
    plt.savefig(saida)
    plt.close()
    
    print(f"Processado (Cinza): {nome}")

print("✅ Tudo pronto no Grayscale!")