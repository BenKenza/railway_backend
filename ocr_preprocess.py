import os
import cv2
import json
import re
import numpy as np
import pytesseract
import logging
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔤 Mot valide = lettre ou chiffre, au moins 3 caractères
def is_valid_word(word):
    return re.fullmatch(r"[A-Za-z0-9éèàâêîôûçÉÈÀÂÊÎÔÛÇ'-]{3,}", word) is not None

# 📚 Charger le dictionnaire médical + français
def load_french_dictionary(txt_path="dict.txt", json_path="MedicalTerms.json"):
    word_set = set()

    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            word_set.update(line.strip().lower() for line in f if len(line.strip()) >= 3)

    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            for section in data:
                for entry in section.get("entries", []):
                    terms = re.findall(r"\b\w+\b", entry.get("term", ""))
                    word_set.update(t.lower() for t in terms if len(t) >= 3)

    return word_set

# 💾 Sauvegarde d'image pour debug
def save_image(img, filename, cmap="gray"):
    os.makedirs("img", exist_ok=True)
    path = os.path.join("img", filename)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    plt.imsave(path, img_rgb, cmap=cmap)

# 🔁 Rotation d’image
def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

# 🔄 Choix de la meilleure rotation
def try_rotations(image, valid_word_set):
    best_img, max_words, best_angle = image, 0, 0

    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(image, angle)
        text = pytesseract.image_to_string(rotated, lang='fra', config='--oem 3 --psm 6')
        words = [w.lower() for w in text.split()]
        valid_words = [w for w in words if is_valid_word(w) and w in valid_word_set]

        logger.info(f"🔄 Angle {angle}° : {len(valid_words)} mots valides")
        if len(valid_words) > max_words:
            best_img, max_words, best_angle = rotated, len(valid_words), angle

    logger.info(f"✅ Meilleure rotation : {best_angle}° avec {max_words} mots")
    return best_img

# 📐 Détection d’inclinaison
def get_skew_angle(image):
    inverted = cv2.bitwise_not(image)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return 0

    angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0]
              if -45 < (theta * 180 / np.pi) - 90 < 45]
    return np.median(angles) if angles else 0

# ↩️ Correction d’inclinaison
def deskew(image):
    angle = get_skew_angle(image)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# 🔍 Pipeline principal
def getmessage(imagefile, debug_mode=True):
    try:
        valid_words_set = load_french_dictionary()

        # Lecture d’image
        if isinstance(imagefile, str):
            img = cv2.imread(imagefile)
        elif isinstance(imagefile, bytes):
            img = cv2.imdecode(np.frombuffer(imagefile, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(imagefile, np.ndarray):
            img = imagefile
        else:
            raise ValueError("❌ Format d'image non supporté")

        if img is None:
            raise ValueError("❌ Image introuvable")

        if debug_mode: save_image(img, "1_original.png")

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if debug_mode: save_image(gray, "2_gray.png")

        rotated = try_rotations(gray, valid_words_set)
        if debug_mode: save_image(rotated, "3_rotated.png")

        deskewed = deskew(rotated)
        if debug_mode: save_image(deskewed, "4_deskewed.png")

        resized = cv2.resize(deskewed, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        if debug_mode: save_image(resized, "5_resized.png")

        thresholded = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 65, 13)
        if debug_mode: save_image(thresholded, "6_thresholded.png")

        # OCR final
        text = pytesseract.image_to_string(thresholded, lang="fra", config='--oem 3 --psm 6')
        logger.info("🧠 OCR terminé")

        if len(text.strip().split()) < 5:
            logger.warning("⚠️ Texte OCR trop court, image peut-être floue")
            return ""

        return text.strip()

    except Exception as e:
        logger.error(f"❌ Erreur dans getmessage : {e}")
        return ""
