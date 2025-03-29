#!/bin/bash

# Initialisation des variables d'environnement
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1  # Réduction de la mémoire

# Lancement de Gunicorn pour exécuter Flask en mode production
exec gunicorn --threads 2 -b 0.0.0.0:8080 app:app
