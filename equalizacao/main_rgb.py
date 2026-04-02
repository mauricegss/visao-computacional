import cv2
import os
import matplotlib.pyplot as plt

input_folder = "assets"
output_folder = "resultados_rgb"

os.makedirs(output_folder, exist_ok=True)

for nome in os.listdir(input_folder):
    caminho = os.path.join(input_folder, nome)
    img_bgr = cv2.imread(caminho)
    
    if img_bgr is None:
        continue
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)
    v_eq = cv2.equalizeHist(v)
    img_hsv_eq = cv2.merge((h, s, v_eq))
    img_eq = cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2RGB)
    
    img_negativo = 255 - img_rgb
    
    plt.figure(figsize=(10,6))
    
    plt.subplot(2,3,1)
    plt.imshow(img_rgb)
    plt.title("Original RGB")
    
    plt.subplot(2,3,2)
    plt.imshow(img_eq)
    plt.title("Equalizada RGB")
    
    plt.subplot(2,3,3)
    plt.imshow(img_negativo)
    plt.title("Negativo RGB")
    
    plt.subplot(2,3,4)
    for i, cor in enumerate(['r', 'g', 'b']):
        plt.hist(img_rgb[:,:,i].ravel(), bins=256, color=cor, alpha=0.5)
    plt.title("Hist Original")
    
    plt.subplot(2,3,5)
    for i, cor in enumerate(['r', 'g', 'b']):
        plt.hist(img_eq[:,:,i].ravel(), bins=256, color=cor, alpha=0.5)
    plt.title("Hist Equalizado")
    
    plt.subplot(2,3,6)
    for i, cor in enumerate(['r', 'g', 'b']):
        plt.hist(img_negativo[:,:,i].ravel(), bins=256, color=cor, alpha=0.5)
    plt.title("Hist Negativo")
    
    plt.tight_layout()
    saida = os.path.join(output_folder, f"resultado_rgb_{nome}.png")
    plt.savefig(saida)
    plt.close()
    
    print(f"Processado RGB: {nome}")