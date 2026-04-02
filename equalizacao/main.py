import cv2
import os
import matplotlib.pyplot as plt

input_folder = "assets-grayscale"
output_folder = "resultados"

os.makedirs(output_folder, exist_ok=True)

for nome in os.listdir(input_folder):
    caminho = os.path.join(input_folder, nome)
    
    img = cv2.imread(caminho, 0)
    
    # Equalização
    equalizada = cv2.equalizeHist(img)
    
    # Negativo
    negativo = 255 - img
    
    # Plot
    plt.figure(figsize=(10,6))
    
    plt.subplot(2,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    
    plt.subplot(2,3,2)
    plt.imshow(equalizada, cmap='gray')
    plt.title("Equalizada")
    
    plt.subplot(2,3,3)
    plt.imshow(negativo, cmap='gray')
    plt.title("Negativo")
    
    plt.subplot(2,3,4)
    plt.hist(img.ravel(), bins=256)
    plt.title("Histograma Original")
    
    plt.subplot(2,3,5)
    plt.hist(equalizada.ravel(), bins=256)
    plt.title("Histograma Equalizado")
    
    plt.subplot(2,3,6)
    plt.hist(negativo.ravel(), bins=256)
    plt.title("Histograma Negativo")
    
    plt.tight_layout()
    
    saida = os.path.join(output_folder, f"resultado_{nome}.png")
    plt.savefig(saida)
    plt.close()
    
    print(f"Processado: {nome}")

print("✅ Tudo pronto!")