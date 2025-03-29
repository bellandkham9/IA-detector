#!/bin/bash

# Initialisation des variables d'environnement
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1  # Réduction de la mémoire

# Exécution de l'application
exec python app.py
