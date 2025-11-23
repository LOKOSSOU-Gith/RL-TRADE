#!/bin/bash

# Script d'installation des dépendances pour RL-trading

echo "=== Installation des dépendances pour RL-trading ==="
echo ""

# Vérifier si on est dans le bon répertoire
if [ ! -f "requirements.txt" ]; then
    echo "Erreur: requirements.txt non trouvé. Assurez-vous d'être dans le répertoire du projet."
    exit 1
fi

# Étape 1: Installer python3-venv et python3-pip si nécessaire
echo "Étape 1: Vérification des prérequis système..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 n'est pas installé. Installation..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
else
    echo "✓ Python3 est installé"
    if ! dpkg -l | grep -q python3-venv; then
        echo "Installation de python3-venv..."
        sudo apt install -y python3-venv python3-pip
    else
        echo "✓ python3-venv est installé"
    fi
fi

echo ""

# Étape 2: Créer un environnement virtuel
echo "Étape 2: Création de l'environnement virtuel..."
if [ -d "venv" ]; then
    echo "L'environnement virtuel existe déjà. Suppression..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Erreur lors de la création de l'environnement virtuel"
    exit 1
fi
echo "✓ Environnement virtuel créé"

echo ""

# Étape 3: Activer l'environnement virtuel et installer les dépendances
echo "Étape 3: Activation de l'environnement virtuel..."
source venv/bin/activate

echo "Mise à jour de pip..."
pip install --upgrade pip

echo ""

# Étape 4: Installer setuptools, pip et wheel aux versions spécifiques
echo "Étape 4: Installation des versions spécifiques de setuptools, pip et wheel..."
pip install setuptools==65.5.0 pip==21 wheel==0.38.0

echo ""

# Étape 5: Installer les dépendances principales
echo "Étape 5: Installation des dépendances depuis requirements.txt..."
pip install -r requirements.txt

echo ""

# Étape 6: Installer gym==0.21.0 (si nécessaire)
echo "Étape 6: Installation de gym==0.21.0..."
pip install gym==0.21.0

echo ""

# Étape 7: Vérifier si rl-baselines3-zoo a un requirements.txt
if [ -f "rl-baselines3-zoo/requirements.txt" ]; then
    echo "Étape 7: Installation des dépendances de rl-baselines3-zoo..."
    pip install -r rl-baselines3-zoo/requirements.txt
else
    echo "Étape 7: rl-baselines3-zoo/requirements.txt non trouvé (optionnel)"
fi

echo ""
echo "=== Installation terminée avec succès! ==="
echo ""
echo "Pour activer l'environnement virtuel, utilisez:"
echo "  source venv/bin/activate"
echo ""
echo "Pour désactiver l'environnement virtuel, utilisez:"
echo "  deactivate"

