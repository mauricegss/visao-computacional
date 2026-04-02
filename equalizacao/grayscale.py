import cv2
import os

# Pastas
input_folder = "assets"
output_folder = "assets-grayscale"

# Cria a pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Extensões suportadas
extensoes = (".jpg", ".webp")

# Percorre todas as imagens
for nome_arquivo in os.listdir(input_folder):
    if nome_arquivo.lower().endswith(extensoes):
        
        caminho_entrada = os.path.join(input_folder, nome_arquivo)
        caminho_saida = os.path.join(output_folder, nome_arquivo)
        
        # Lê imagem colorida
        img = cv2.imread(caminho_entrada)
        
        # Converte para grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Salva imagem
        cv2.imwrite(caminho_saida, gray)
        
        print(f"Convertido: {nome_arquivo}")

print("✅ Todas as imagens foram convertidas!")