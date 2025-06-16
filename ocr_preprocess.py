"""import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def getmessage(imagefile, debug_mode=True):
    try:
        logger.info("🟢 Début getmessage")

        if isinstance(imagefile, str):
            logger.info("🖼️ Chargement de l'image depuis le chemin : %s", imagefile)
            img = cv2.imread(imagefile)
        elif isinstance(imagefile, bytes):
            logger.info("🧾 Chargement de l'image depuis bytes")
            nparr = np.frombuffer(imagefile, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(imagefile, np.ndarray):
            logger.info("📦 Chargement depuis tableau NumPy")
            img = imagefile
        else:
            raise ValueError("❌ Format d'image non supporté")

        if img is None:
            raise ValueError("❌ Impossible de charger l'image")

        logger.info("📷 Conversion en niveaux de gris")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        logger.info("🔍 Agrandissement")
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        logger.info("🧪 Seuillage")
        thresholded = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 65, 13
        )

        logger.info("🧠 Lancement de Tesseract OCR")
        text = pytesseract.image_to_string(thresholded, lang='fra', config='--oem 3 --psm 6')
        logger.info("✅ OCR terminé")

        return text.strip()

    except Exception as e:
        logger.error(f"❌ Erreur dans getmessage : {str(e)}")
        raise
"""
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import os
import json
import logging
import re

# 📋 Initialisation du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Vérifie si un mot est valide (lettres/chiffres, au moins 2 caractères)
def is_valid_word(word):
    return re.fullmatch(r"[A-Za-z0-9éèàâêîôûçÉÈÀÂÊÎÔÛÇ'-]{2,}", word) is not None

# 📚 Charge les dictionnaires (dict.txt et MedicalTerms.json)
def load_french_dictionary(txt_path="dict.txt", json_path="MedicalTerms.json"):
    word_set = set()

    # Charger les mots du fichier texte
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                word_set.add(line.strip().lower())

    # Charger les termes du JSON médical
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            for section in data:
                for entry in section.get("entries", []):
                    term = entry.get("term", "")
                    for t in re.findall(r"\b\w+\b", term):
                        word_set.add(t.lower())

    return word_set

# 📁 Enregistre une image dans le dossier 'img'
def save_image(img, filename, cmap="gray"):
    output_dir = "img"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    if len(img.shape) == 2:
        plt.imsave(path, img, cmap=cmap)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave(path, img_rgb)

# 🔁 Fait pivoter une image
def rotate_image(image, angle):
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# 🔄 Essaie plusieurs rotations et garde celle avec le plus de mots français corrects
def try_rotations(image, valid_word_set):
    best_img = image
    max_words = 0
    best_angle = 0

    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(image, angle)
        text = pytesseract.image_to_string(rotated, lang='fra', config='--oem 3 --psm 6')
        words = text.split()
        valid_words = [w for w in words if is_valid_word(w) and w.lower() in valid_word_set]

        logger.info(f"🔄 Angle {angle}° : {len(valid_words)} mots valides détectés")
        logger.info(f"Mots : {valid_words}")

        if len(valid_words) > max_words:
            max_words = len(valid_words)
            best_img = rotated
            best_angle = angle

    logger.info(f"✅ Meilleure rotation : {best_angle}°, avec {max_words} mots valides")
    return best_img

# 📐 Détection et correction de l'inclinaison
def get_skew_angle(image):
    inverted = cv2.bitwise_not(image)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return 0

    angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0] if -45 < (theta * 180 / np.pi) - 90 < 45]

    return np.median(angles) if angles else 0

def deskew(image):
    angle = get_skew_angle(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# 🔍 Pipeline complet
def getmessage(imagefile, debug_mode=True):
    valid_words_set = load_french_dictionary()

    try:
        if isinstance(imagefile, str):
            img = cv2.imread(imagefile)
        elif isinstance(imagefile, bytes):
            nparr = np.frombuffer(imagefile, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(imagefile, np.ndarray):
            img = imagefile
        else:
            raise ValueError("Format d'image non supporté")

        if img is None:
            raise ValueError("Image introuvable")

        if debug_mode:
            save_image(img, "1. Image originale.png")
            logger.info("Image originale chargée")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if debug_mode:
            save_image(gray, "2. Gray.png")

        rotated = try_rotations(gray, valid_words_set)
        """if debug_mode:
            save_image(rotated, "3. Rotated.png")

        deskewed = deskew(rotated)"""
        """if debug_mode:
            save_image(deskewed, "4. Deskewed.png")

        resized = cv2.resize(deskewed, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)"""
        if debug_mode:
            save_image(resized, "5. Resized.png")

        thresholded = cv2.adaptiveThreshold(
            rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 65, 13
        )
        if debug_mode:
            save_image(thresholded, "6. Thresholded.png")

        text = pytesseract.image_to_string(thresholded, lang="fra", config='--oem 3 --psm 6')
        logger.info("OCR terminé.")
        return text.strip()

    except Exception as e:
        logger.error(f"Erreur : {str(e)}")
        raise
