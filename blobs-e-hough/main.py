import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

input_folder = "./assets"
output_folder = "./resultados"
os.makedirs(output_folder, exist_ok=True)

for nome in os.listdir(input_folder):
    caminho = os.path.join(input_folder, nome)
    img = cv2.imread(caminho)
    if img is None: continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    
    # --- 1. CÍRCULOS ---
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=100,
        param1=100, param2=30,
        minRadius=10, maxRadius=400
    )
    
    img_circles = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    # --- 2. LINHAS (EXTRA) ---
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    
    img_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # --- IMAGEM 1: ORIGINAL + CÍRCULOS ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB))
    plt.title("Hough Círculos")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"circulos_{nome}"))
    plt.close()

    # --- IMAGEM 2: LINHAS (EXTRA) ---
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.title("Hough Linhas (Extra)")
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"linhas_{nome}"))
    plt.close()

print("✅ Experimentos separados concluídos!")