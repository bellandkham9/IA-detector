import os
import torch
from flask import Flask, request, jsonify
import timm
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import logging
from werkzeug.utils import secure_filename


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================
# CONFIGURATIONS & OPTIMISATIONS
# ==============================================

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Désactive le GPU (optionnel)
os.environ["OMP_NUM_THREADS"] = "1"  # Réduction de la consommation mémoire
os.environ['PYTHONWARNINGS'] = 'ignore'  # Ignore les warnings inutiles

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limite upload 50MB

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Détection automatique du CPU/GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device set to use {DEVICE}")

# ==============================================
# INITIALISATION DES MODÈLES (Lazy Loading)
# ==============================================

def get_image_model():
    """Chargement paresseux du modèle EfficientNet pour éviter de surcharger la RAM"""
    logger.info("Chargement du modèle image...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2).to(DEVICE)
    model.eval()
    return model

def get_nlp_model():
    """Chargement paresseux du modèle NLP"""
    logger.info("Chargement du modèle NLP...")
    return pipeline("text-classification", model="distilbert-base-uncased")

# ==============================================
# FONCTIONS UTILITAIRES
# ==============================================

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Prétraitement des images pour le modèle EfficientNet"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ajustement à EfficientNet B0
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

def extract_text_from_url(url: str) -> str:
    """Extraction du contenu textuel d'une page web"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Suppression des balises inutiles
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()

        text = '\n'.join([p.get_text(strip=True) for p in soup.find_all('p')])
        return text if len(text) > 100 else None
    except Exception as e:
        logger.error(f"Erreur d'extraction : {e}")
        return None

# ==============================================
# ROUTES API
# ==============================================

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Analyse d'une image pour détection IA"""
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    try:
        file = request.files['file']
        img = Image.open(file.stream)
        tensor = preprocess_image(img)

        model = get_image_model()
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            ai_score = probs[0][1].item()

        return jsonify({"label": "GENERATED" if ai_score > 0.7 else "REAL", "score": ai_score, "success": True})
    except Exception as e:
        logger.error(f"Erreur analyse image: {e}")
        return jsonify({"error": str(e), "success": False}), 500


# Assure-toi que ces fonctions existent et sont définies ailleurs dans ton code
def preprocess_frame(frame):
    """Prétraitement d'une frame pour le modèle"""
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(frame).unsqueeze(0)  # Ajout d'une dimension batch

# Charger le modèle une seule fois pour éviter les rechargements inutiles
def load_image_model():
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    model.eval()
    return model

image_model = load_image_model()

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    """Analyse d'une vidéo pour détection IA"""
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "Fichier sans nom"}), 400

    temp_path = secure_filename(f"temp_{file.filename}")
    file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        os.remove(temp_path)
        return jsonify({"error": "Vidéo vide ou corrompue"}), 400

    frame_indices = np.linspace(0, total_frames-1, min(30, total_frames), dtype=int)
    scores = []

    try:
        with torch.no_grad():  # Économie de mémoire et meilleure performance
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    try:
                        tensor = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        outputs = image_model(tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        scores.append(probs[0][1].item())
                    except Exception as e:
                        logger.warning(f"Erreur frame {idx}: {e}")

        cap.release()
        os.remove(temp_path)

        if not scores:
            return jsonify({"error": "Aucune frame analysable", "success": False}), 500

        avg_score = np.mean(scores)
        return jsonify({
            "label": "GENERATED" if avg_score > 0.7 else "REAL",
            "score": avg_score,
            "frames_analyzed": len(scores),
            "success": True
        })

    except Exception as e:
        logger.error(f"Erreur analyse vidéo: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e), "success": False}), 500



@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyse d'URL pour détection de fake news"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "URL requise", "success": False}), 400

    try:
        text = extract_text_from_url(data['url'])
        if not text:
            return jsonify({"error": "Impossible d'extraire le contenu", "success": False}), 400

        model = get_nlp_model()
        result = model(text[:512])[0]  # Limite à 512 tokens
        return jsonify({"label": result['label'], "score": float(result['score']), "success": True})
    except Exception as e:
        logger.error(f"Erreur analyse texte: {e}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/')
def home():
    return jsonify({"message": "Bienvenue sur mon API Flask optimisée !"})

# ==============================================
# DÉMARRAGE DU SERVEUR
# ==============================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
