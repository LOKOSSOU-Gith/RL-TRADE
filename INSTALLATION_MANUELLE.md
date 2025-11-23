# Guide d'Installation Manuelle

## Option 1 : Installation avec sudo (Recommandé)

Exécutez ces commandes dans votre terminal :

```bash
cd "/media/gryphen/Disque local/SERIE/LINUX/RL-trading-main"

# 1. Installer les prérequis système
sudo apt update
sudo apt install -y python3-venv python3-pip

# 2. Créer l'environnement virtuel
python3 -m venv venv

# 3. Activer l'environnement virtuel
source venv/bin/activate

# 4. Mettre à jour pip
pip install --upgrade pip

# 5. Installer les versions spécifiques requises
pip install setuptools==65.5.0 pip==21 wheel==0.38.0

# 6. Installer les dépendances principales
pip install -r requirements.txt

# 7. Installer gym==0.21.0
pip install gym==0.21.0

# 8. (Optionnel) Installer les dépendances de rl-baselines3-zoo si le fichier existe
if [ -f "rl-baselines3-zoo/requirements.txt" ]; then
    pip install -r rl-baselines3-zoo/requirements.txt
fi
```

## Option 2 : Installation sans sudo (avec --user)

Si vous ne pouvez pas utiliser sudo, vous pouvez installer les packages pour l'utilisateur :

```bash
cd "/media/gryphen/Disque local/SERIE/LINUX/RL-trading-main"

# 1. Installer pip pour l'utilisateur (si pas déjà installé)
python3 -m ensurepip --user || curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py --user

# 2. Ajouter ~/.local/bin au PATH (ajoutez cette ligne à votre ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"

# 3. Installer les versions spécifiques
python3 -m pip install --user setuptools==65.5.0 pip==21 wheel==0.38.0

# 4. Installer les dépendances
python3 -m pip install --user -r requirements.txt

# 5. Installer gym
python3 -m pip install --user gym==0.21.0
```

⚠️ **Note** : L'option 2 peut avoir des problèmes de compatibilité. L'option 1 avec un environnement virtuel est fortement recommandée.

## Vérification de l'installation

Après l'installation, testez avec :

```bash
# Si vous utilisez un environnement virtuel
source venv/bin/activate

# Tester l'import
python3 -c "import stable_baselines3; import gym; print('Installation réussie!')"
```

## Utilisation future

**Avec environnement virtuel :**
```bash
source venv/bin/activate
# Votre code ici
deactivate  # Pour désactiver
```

**Sans environnement virtuel (installation --user) :**
```bash
# Utilisez directement python3
python3 votre_script.py
```

