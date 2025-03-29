# Utilise une image Python officielle et légère
FROM python:3.9-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    python3-dev \
    libgl1-mesa-glx libsm6 libxrender1 libxext6 ffmpeg \
    build-essential pkg-config

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet
COPY . .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 8080 (Railway peut rediriger vers ce port)
EXPOSE 8080

# Commande de démarrage avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--threads", "2", "--timeout", "300", "app:app"]
