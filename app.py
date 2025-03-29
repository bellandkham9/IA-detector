import os
os.system('apt-get update && apt-get install -y libgl1')
from flask import Flask, request, jsonify
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import logging
from typing import Optional, Dict, Tuple

# Configuration de l'application Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limite de 50MB pour les uploads

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================
# INITIALISATION DES MODÈLES
# ==============================================

# Modèle de détection d'images
try:
    image_model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=2)
    image_model.eval()
    logger.info("Modèle de détection d'images chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle image: {e}")
    raise

# Modèle de détection de fake news
try:
    fake_news_model = pipeline("text-classification", model="roberta-base-openai-detector")
    logger.info("Modèle de détection de fake news chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle NLP: {e}")
    raise

# ==============================================
# FONCTIONS UTILITAIRES
# ==============================================

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Prétraitement des images pour le modèle"""
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Prétraitement des frames vidéo"""
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(Image.fromarray(frame)).unsqueeze(0)

def extract_text_from_url(url: str) -> Optional[str]:
    """Extraction du contenu textuel d'une URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()

        article = soup.find('article') or soup.find('main') or soup.body
        if article:
            paragraphs = article.find_all(['p', 'h1', 'h2', 'h3'])
            text = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return text if len(text) > 100 else None

        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction: {e}")
        return None

# ==============================================
# ROUTES DE L'API
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

        with torch.no_grad():
            outputs = image_model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            ai_score = probs[0][1].item()

        return jsonify({
            "label": "GENERATED" if ai_score > 0.7 else "REAL",
            "score": ai_score,
            "success": True
        })

    except Exception as e:
        logger.error(f"Erreur analyse image: {e}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    """Analyse d'une vidéo pour détection IA"""
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    try:
        file = request.files['file']
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)

        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, min(30, total_frames), dtype=int)

        scores = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                try:
                    tensor = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
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
            return jsonify({
                "error": "Impossible d'extraire le contenu",
                "success": False
            }), 400

        result = fake_news_model(text[:512])[0]  # Limite à 512 tokens
        return jsonify({
            "label": result['label'],
            "score": float(result['score']),
            "success": True
        })

    except Exception as e:
        logger.error(f"Erreur analyse texte: {e}")
        return jsonify({"error": str(e), "success": False}), 500

# ==============================================
# POINT D'ENTRÉE
# ==============================================


@app.route('/')
def home():
    return jsonify({"message": "Bienvenue sur mon API Flask déployée sur Vercel !"})

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({"message": "Voici ta prédiction !"})

# Assurez-vous que le serveur Flask s'exécute bien avec Vercel
def handler(event, context):
    return app(event, context)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)