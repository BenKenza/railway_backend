# Image légère avec Python
FROM python:3.11-slim

# Installer Tesseract OCR + langue française + dépendances
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Créer un dossier de travail
WORKDIR /app

# Copier tout le projet dans le conteneur
COPY . .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Définir la commande par défaut (ici Flask ou Gunicorn)
CMD ["python", "app.py"]

# Exposer le port Flask
EXPOSE 5000
