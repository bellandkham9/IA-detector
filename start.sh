#!/bin/bash
PORT=${PORT:-8000}  # Valeur par défaut 8000 si $PORT non défini
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300 app:app