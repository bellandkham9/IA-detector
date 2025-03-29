# Utiliser une image Python officielle
FROM python:3.9-slim-bullseye

# Installer les dépendances
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libsm6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Solution clé - Utilisez un script de démarrage
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]