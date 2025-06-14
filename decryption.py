import os
import sys
import re
import json
import ocr_preprocess
import google.generativeai as genai
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configurer la clé API Gemini

genai.configure(api_key=os.getenv("GEMINI_API_KEY")) # Remplace par ta vraie clé API

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")


if len(sys.argv) != 5:
    print("Usage : python script.py <image_path> <gender> <age> <smoking>")
    sys.exit(1)

image_path = sys.argv[1]
gender = sys.argv[2]
age = sys.argv[3]
smoking = sys.argv[4]
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


logger.info(f"Décryptage de l'image : {image_path}, gender={gender}, age={age}, smoking={smoking}")

message = ocr_preprocess.getmessage(image_path)
logger.info(f"Texte OCR extrait : {message}")
message = message.replace('\n', ' ')
message = re.sub(r'\d{2}-\d{2}-\d{2,4}', '', message)
if not message.strip():
    print(json.dumps({
        "error": "Le texte OCR est vide. Merci de reprendre une image plus claire."
    }))
    sys.exit(0)
if message:
    sex_num = 1 if gender.lower() in ['homme', 'male', 'masculin'] else 0
    smoking_num = 1 if smoking.lower() in ['oui', 'yes', 'true', 'smoker'] else 0



    prompt = f"""
Tu es un assistant médical expert.

Je te fournis un texte brut OCR contenant des résultats d’analyses médicales. Ce texte peut inclure :
- des noms d’analyses (ex. : Hémoglobine, Créatinine, etc.),
- des valeurs (ex. : 11.5 g/dL, 1.2 mg/dL),
- mais aussi des données non pertinentes (noms du laboratoire, informations administratives, détails du patient, etc.).

Voici le texte brut :
{message}


Informations supplémentaires :
- Le sexe du patient est déjà connu : - Sexe: {sex_num}
- L'âge du patient est déjà connu : - Âge: {age} ans.
- Fumeur: {smoking_num}.
Ta mission :

1. Ignore tout le texte qui n’est pas une analyse médicale (exemple : nom du labo, informations patient, date, etc.).
2. Associe chaque nom d’analyse à sa valeur correspondante même si le format est perturbé.
3. Pour chaque test médical identifié :
   - Indique pour chaque test :
   - son identifiant (nom du test),
   - sa valeur,
   - son unité,
   - sa plage de référence médicale (exemple : pour Hémoglobine → "12–16 g/dL"),
   - une interprétation : 'bad', 'normal' ou 'illogical'.

- Structure chaque test sur une ligne au format JSON suivant :
{{"identifiant": "nom_du_test", "value": 45, "measurement": "ml", "reference": "plage_attendue", "interpretation": "bad"}}


4. Analyse uniquement les résultats pertinents pour remplir les colonnes suivantes (comme dans un fichier CSV médical) :

["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]
5. Si l'unité extraite ne correspond pas à celle de mon dataset, convertis la valeur dans l'unité attendue. Les unités attendues sont les suivantes :

age : années (ans)
anaemia : pas d'unité (valeur binaire)
creatinine_phosphokinase : UI/L (Unité Internationale par litre)
diabetes : pas d'unité (valeur binaire)
ejection_fraction : pourcentage (%)
high_blood_pressure : pas d'unité (valeur binaire)
platelets : cellules/µL (plaquettes par microlitre)
serum_creatinine : mg/dL (milligrammes par décilitre)
serum_sodium : mEq/L (milléquivalents par litre)
sex : pas d'unité (valeur binaire)
smoking : pas d'unité (valeur binaire)
time : mois
Règles de remplissage :

1. **Hématologie (NFS - Numération Formule Sanguine)** :
   - Si l'hémoglobine est inférieure à 12 g/dL, alors **anaemia** = 1, sinon 0.
   - **platelets** : Si la numération plaquettaire est présente, utilise la valeur pour ce champ.

2. **Biochimie Sanguine** :
   - **serum_creatinine** : Utilise le niveau de créatinine pour remplir ce champ.
   - **diabetes** : Si le glucose à jeun est supérieur à 1.26 g/L, alors **diabetes** = 1.
   - **serum_sodium** : Remplis ce champ avec la valeur du sodium dans le sang.

3. **Bilan Cardiaque** :
   - **creatinine_phosphokinase** : Si la CPK (créatine phosphokinase) est présente, utilise cette valeur pour le champ **creatinine_phosphokinase**.


Autres règles :
- **sex** = 1 si homme, sinon 0.

Retourne un objet JSON strictement valide avec deux sections : "results" et "data". Ne mets aucun commentaire (pas de // ou /* ... */). Le JSON doit être parfaitement décodable sans aucune explication dans les valeurs. Toute explication doit être faite hors du JSON si nécessaire.

1. **results** : Liste des tests avec leurs identifiants, valeurs, unités et interprétations sous forme d'objets JSON.
2. **data** : Un objet contenant les résultats agrégés pour remplir les colonnes du fichier CSV, y compris les champs comme `"anaemia"`, `"creatinine_phosphokinase"`, `"diabetes"`, `"platelets"`, etc., avec les valeurs interprétées.
"""

    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()
        # 🟡 Ajoute cette partie ici
        if ('"results"' not in reply and '"data"' not in reply):
            print(json.dumps({
                "error": "Le texte OCR semble incompréhensible. Merci de reprendre une image plus lisible."
            }))
            sys.exit(0)
        # Nettoyer automatiquement les balises Markdown inutiles
        if reply.startswith('```json'):
            reply = reply.replace('```json', '').strip()
        elif reply.startswith("'''json"):
            reply = reply.replace("'''json", '').strip()
        if reply.endswith('```') or reply.endswith("'''"):
            reply = reply[:-3].strip()

        try:
            json_data = json.loads(reply)

            # ➕ Remplissage automatique des champs manquants avec moyennes
            #if "data" in json_data:
            #    json_data["data"] = remplir_champs_vides_par_moyenne(json_data["data"])

            print(json.dumps(json_data, indent=2, ensure_ascii=False))

        except json.JSONDecodeError:
            print("Réponse reçue mais le JSON n'est pas valide :\n")
            print(reply)

    except Exception as e:
        print(f"Erreur lors de l'appel à Gemini : {e}")
