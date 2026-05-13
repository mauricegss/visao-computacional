import cv2
import numpy as np
import os

# =========================================================
# CONFIGURAÇÕES
# =========================================================

VIDEO_PATH = "video.mp4"   # coloque o nome do vídeo aqui
OUTPUT_DIR = "resultados"

# quantidade de frames usados para construir os modelos
NUM_FRAMES_MODELO = 100

# frames que serão salvos no relatório
FRAMES_RELATORIO = [40, 80]

# threshold da segmentação
THRESHOLD = 30

# =========================================================
# CRIA PASTAS
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

pastas = [
    "original",
    "fundo_fixo",
    "media",
    "mediana",
    "rgb_media",
]

for p in pastas:
    os.makedirs(os.path.join(OUTPUT_DIR, p), exist_ok=True)

# =========================================================
# LEITURA DO VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

frames_gray = []
frames_rgb = []

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frames_gray.append(gray)
    frames_rgb.append(frame)

    count += 1

    if count >= NUM_FRAMES_MODELO:
        break

cap.release()

frames_gray = np.array(frames_gray)
frames_rgb = np.array(frames_rgb)

print("Frames carregados:", len(frames_gray))

# =========================================================
# MODELO FUNDO FIXO
# =========================================================

fundo_fixo = frames_gray[0]

# =========================================================
# MODELO MÉDIA
# =========================================================

modelo_media = np.mean(frames_gray, axis=0).astype(np.uint8)

# =========================================================
# MODELO MEDIANA
# =========================================================

modelo_mediana = np.median(frames_gray, axis=0).astype(np.uint8)

# =========================================================
# EXTRA RGB (MÉDIA RGB)
# =========================================================

modelo_media_rgb = np.mean(frames_rgb, axis=0).astype(np.uint8)

# =========================================================
# PROCESSAMENTO DOS FRAMES DO RELATÓRIO
# =========================================================

for idx in FRAMES_RELATORIO:

    frame_gray = frames_gray[idx]
    frame_rgb = frames_rgb[idx]

    # -----------------------------------------------------
    # FUNDO FIXO
    # -----------------------------------------------------

    diff_fixo = cv2.absdiff(frame_gray, fundo_fixo)

    _, mask_fixo = cv2.threshold(
        diff_fixo,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------------------------------
    # MÉDIA
    # -----------------------------------------------------

    diff_media = cv2.absdiff(frame_gray, modelo_media)

    _, mask_media = cv2.threshold(
        diff_media,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------------------------------
    # MEDIANA
    # -----------------------------------------------------

    diff_mediana = cv2.absdiff(frame_gray, modelo_mediana)

    _, mask_mediana = cv2.threshold(
        diff_mediana,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------------------------------
    # RGB EXTRA
    # -----------------------------------------------------

    diff_rgb = cv2.absdiff(frame_rgb, modelo_media_rgb)

    # threshold em cada canal
    b, g, r = cv2.split(diff_rgb)

    _, b = cv2.threshold(b, THRESHOLD, 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, THRESHOLD, 255, cv2.THRESH_BINARY)
    _, r = cv2.threshold(r, THRESHOLD, 255, cv2.THRESH_BINARY)

    mask_rgb = cv2.merge([b, g, r])

    # =====================================================
    # SALVANDO IMAGENS
    # =====================================================

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "original", f"original_frame_{idx}.png"),
        frame_gray
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "fundo_fixo", f"fixo_frame_{idx}.png"),
        mask_fixo
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "media", f"media_frame_{idx}.png"),
        mask_media
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "mediana", f"mediana_frame_{idx}.png"),
        mask_mediana
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "rgb_media", f"rgb_frame_{idx}.png"),
        mask_rgb
    )

print("Resultados salvos em:", OUTPUT_DIR)