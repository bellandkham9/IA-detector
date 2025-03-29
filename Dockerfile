FROM python:3.9-slim-bullseye

# 1. Installe les dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libsm6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Copie le script de démarrage en premier (optimisation du cache)
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 3. Installe les dépendances Python
WORKDIR /app
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copie le code applicatif
COPY . . 

# 5. Exécute le script
# CMD ["/start.sh"]  # <-- Utilise le chemin absolu
CMD ["gunicorn", "--threads", "2", "-b", "0.0.0.0:8080", "app:app"]

