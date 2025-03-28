import requests
import os
from typing import Dict, Any

# Configuration de base
API_BASE_URL = "http://localhost:5000"  # Remplacez par l'URL de votre API
TEST_IMAGE_PATH = "DSC02687.jpg"
TEST_VIDEO_PATH = "amour.mp4"
TEST_URLS = [
    "https://fr.wikipedia.org/wiki/Terre",
    "https://example.com/fake-article"
]

def test_image_analysis(image_path: str) -> Dict[str, Any]:
    """Teste l'endpoint /api/analyze/image"""
    try:
        with open(image_path, 'rb') as img_file:
            response = requests.post(
                f"{API_BASE_URL}/api/analyze/image",
                files={'file': img_file}
            )
        return response.json()
    except Exception as e:
        return {"error": str(e), "success": False}

def test_video_analysis(video_path: str, max_frames: int = 30) -> Dict[str, Any]:
    """Teste l'endpoint /api/analyze/video"""
    try:
        with open(video_path, 'rb') as video_file:
            response = requests.post(
                f"{API_BASE_URL}/api/analyze/video",
                files={'file': video_file},
                data={'max_frames': max_frames}  # Si votre API supporte ce paramètre
            )
        return response.json()
    except Exception as e:
        return {"error": str(e), "success": False}

def test_text_analysis(url: str) -> Dict[str, Any]:
    """Teste l'endpoint /api/analyze/text"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/analyze/text",
            json={'url': url}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e), "success": False}

def main():
    print("=== Début des tests ===")
    
    # 1. Test d'analyse d'image
    if os.path.exists(TEST_IMAGE_PATH):
        print("\n[1/3] Test d'analyse d'image...")
        result = test_image_analysis(TEST_IMAGE_PATH)
        print(f"Résultat image : {result.get('label')} (score: {result.get('score', 0):.2f})")
    else:
        print(f"\n[1/3] Fichier image introuvable : {TEST_IMAGE_PATH}")
    
    # 2. Test d'analyse vidéo
    if os.path.exists(TEST_VIDEO_PATH):
        print("\n[2/3] Test d'analyse vidéo...")
        result = test_video_analysis(TEST_VIDEO_PATH)
        print(f"Résultat vidéo : {result.get('label')} (score moyen: {result.get('score', 0):.2f}, {result.get('frames_analyzed', 0)} frames analysées)")
    else:
        print(f"\n[2/3] Fichier vidéo introuvable : {TEST_VIDEO_PATH}")
    
    # 3. Test d'analyse de texte
    print("\n[3/3] Test d'analyse de texte (fake news)...")
    for url in TEST_URLS:
        result = test_text_analysis(url)
        if result.get('success'):
            print(f"\nRésultats pour {url}:")
            print(f"Label: {result['label']}, Score: {result['score']:.4f}")
            print(f"Longueur du texte: {result.get('text_length', 'N/A')} caractères")
        else:
            print(f"\nErreur pour {url}: {result.get('error', 'Erreur inconnue')}")
    
    print("\n=== Tests terminés ===")

if __name__ == "__main__":
    main()