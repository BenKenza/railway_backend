import os
import sys
import re
import json
import logging
import textwrap
from dotenv import load_dotenv
import ocr_preprocess
import google.generativeai as genai

# 🛠️ Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 🔐 Charger les variables d'environnement
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# ✅ Vérifier les arguments
if len(sys.argv) != 5:
    print("Usage : python script.py <image_path> <gender> <age> <smoking>")
    sys.exit(1)

image_path, gender, age, smoking = sys.argv[1:5]
logger.info(f"📥 Décryptage de : {image_path} | Genre: {gender} | Âge: {age} | Fumeur: {smoking}")

# 🔍 OCR
message = ocr_preprocess.getmessage(image_path)
message = re.sub(r'\n', ' ', message)
print('text: ',message);
message = re.sub(r'\d{2}-\d{2}-\d{2,4}', '', message)

if not message.strip():
    print(json.dumps({"error": "Le texte OCR est vide. 111111Merci de reprendre une image plus claire."}))
    sys.exit(0)

# 🔢 Encodage des infos patient
sex_num = 1 if gender.lower() in ['homme', 'male', 'masculin'] else 0
smoking_num = 1 if smoking.lower() in ['oui', 'yes', 'true', 'smoker'] else 0

# 🧠 Prompt pour Gemini
prompt = f"""
Tu es un assistant médical expert. Voici un texte brut OCR contenant des résultats d’analyses médicales :

{textwrap.shorten(message, width=3000)}  # Pour éviter les prompts trop longs

Informations patient :
- Sexe: {sex_num} (1=homme, 0=femme)
- Âge: {age} ans
- Fumeur: {smoking_num}

Ta mission :
1. Ignore les données non médicales ou personnelles.
2. Identifie chaque test médical et associe-le à :
   - sa valeur (`value`)
   - son unité correcte (`measurement`) — si l'unité est absente ou incohérente, déduis-la selon le test.
   - sa plage de référence (`reference`) 
   - une interprétation (`interpretation`) parmi : `"normal"`, `"bad"`, `"illogical"`

3. Structure chaque test sous ce format JSON :
{{
  "identifiant": "nom_du_test",
  "value": 45,
  "measurement": "ml",
  "reference": "plage_attendue",
  "interpretation": "bad"
}}

4. Remplis uniquement les champs du dataset suivant :
["age", "creatinine_phosphokinase", "ejection_fraction", "sex"]
- creatinine_phosphokinase : valeur en UI/L
- ejection_fraction : en %
- age et sex sont déjà fournis

Exigences :
- Corrige toute incohérence dans l’unité ou la plage de référence selon des normes médicales.
- Si une information est absente, complète-la avec des valeurs médicales standard.
- Retourne un objet JSON **strictement valide**, contenant :
  - `"results"` : liste des tests analysés
  - `"data"` : dictionnaire avec les valeurs pour le dataset

Réponds uniquement avec cet objet JSON.
"""


# 🚀 Appel à Gemini
try:
    response = model.generate_content(prompt)
    reply = response.text.strip()

    # Nettoyage des balises Markdown éventuelles
    for md in ['```json', "'''json", '```', "'''"]:
        reply = reply.replace(md, '').strip()

    if '"results"' not in reply or '"data"' not in reply:
        print(json.dumps({"error": "Le texte OCR semble incompréhensible. Merci de reprendre une image plus lisible."}))
        sys.exit(0)

    json_data = json.loads(reply)
    print(json.dumps(json_data, indent=2, ensure_ascii=False))

except json.JSONDecodeError:
    """print("Réponse Gemini invalide :")"""
    print(reply)
except Exception as e:
    print(f"Erreur lors de l'appel à Gemini : {e}")
