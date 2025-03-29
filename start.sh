#!/bin/bash
# Active l'environnement Python (si nécessaire)
source /opt/venv/bin/activate 2>/dev/null || true

# Définit le port par défaut
PORT=${PORT:-8000}

# Lance Gunicorn avec la configuration optimisée
exec gunicorn --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    app:app