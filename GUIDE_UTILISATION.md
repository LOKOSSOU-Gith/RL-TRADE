# Guide d'Utilisation - Bot de Trading par Reinforcement Learning

## 📋 Vue d'ensemble

Ce projet est un **bot de trading automatisé** utilisant le **Reinforcement Learning (RL)** pour trader sur le marché Forex, spécifiquement la paire **EUR/USD**. Il utilise des algorithmes d'apprentissage par renforcement comme **PPO**, **DQN**, et **A2C** pour apprendre des stratégies de trading rentables.

## 🚀 Installation

### 1. Prérequis
Assurez-vous d'avoir Python installé (version 3.7+ recommandée).

### 2. Installation des dépendances

```bash
# Installer les versions spécifiques de setuptools et pip (nécessaire pour gym==0.21.0)
pip install setuptools==65.5.0 pip==21 
pip install wheel==0.38.0

# Installer les dépendances principales
pip install -r requirements.txt

# Installer les dépendances de rl-baselines3-zoo (si le sous-module est présent)
pip install -r rl-baselines3-zoo/requirements.txt
```

### 3. Structure du projet
```
RL-trading-main/
├── config.py              # Configuration des chemins (DATA_PATH, LOGS_PATH, etc.)
├── requirements.txt       # Dépendances Python
├── hyperparams/          # Hyperparamètres pour les algorithmes RL
│   ├── default/          # Hyperparamètres par défaut
│   └── tuned/            # Hyperparamètres optimisés
├── rl_trading/           # Code principal du bot
│   ├── data/             # Gestion des données Forex
│   ├── environments/     # Environnements de trading (Gym)
│   └── utils/            # Utilitaires
├── notebooks/            # Notebooks Jupyter pour expérimenter
└── illustrations/        # Graphiques de résultats
```

## 📊 Comment utiliser le bot

### Méthode 1 : Utilisation via les Notebooks Jupyter (Recommandé pour débuter)

Les notebooks sont la meilleure façon de comprendre et d'utiliser le bot :

1. **Test de l'environnement** : `notebooks/forex_environment_test.ipynb`
   - Teste l'environnement de trading de base
   - Montre comment créer un environnement simple

2. **Expériences RL complètes** : `notebooks/forex_full_eurusd_rl_experiments.ipynb`
   - Entraîne des modèles RL (PPO, DQN, A2C)
   - Effectue l'optimisation des hyperparamètres
   - Évalue les performances

3. **Analyse des meilleurs modèles** : `notebooks/forex_full_eurusd_best_rl_models_analysis.ipynb`
   - Analyse les résultats des modèles entraînés
   - Visualise les performances

### Méthode 2 : Utilisation programmatique (Python)

#### Exemple 1 : Créer un environnement de trading simple

```python
import pandas as pd
from rl_trading.environments import (
    Actions,
    ForexEnvBasic,
    ForexMarketOrderStrategyAllIn,
    ForexRewardStrategyLogPortfolioReturn,
    ForexTradingCostsStrategySpread
)
from rl_trading.data.forex import (
    ForexDataSource,
    load_processed_forex_data,
)
from config import DATA_PATH

# Charger les données Forex
forex_data = load_processed_forex_data(
    DATA_PATH, 
    ForexDataSource.FOREXTESTER, 
    pairs=['EURUSD'], 
    version='Agg'
)

# Créer l'environnement de trading
env = ForexEnvBasic(
    target_prices_df=forex_data['EURUSD'],
    features_df=forex_data['EURUSD'].drop('<DT>', axis=1),
    portfolio_value=1000,  # Capital initial
    allowed_actions={Actions.SELL, Actions.CLOSE, Actions.BUY},
    market_order_strategy=ForexMarketOrderStrategyAllIn(),
    reward_strategy=ForexRewardStrategyLogPortfolioReturn(),
    trading_costs_strategy=ForexTradingCostsStrategySpread(spread=0.0001),
    include_in_obs=['position']
)

# Tester l'environnement
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Action aléatoire (pour test)
    obs, reward, done, info = env.step(action)
    if done:
        break

# Visualiser les résultats
env.render()
```

#### Exemple 2 : Entraîner un modèle PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_trading.environments import (
    Actions,
    ForexEnvBasic,
    ForexMarketOrderStrategyAllIn,
    ForexRewardStrategyLogPortfolioReturn,
    ForexTradingCostsStrategySpread
)

# Créer l'environnement (comme dans l'exemple 1)
def make_env():
    # ... code de création d'environnement ...
    return env

# Créer un environnement vectorisé
vec_env = DummyVecEnv([make_env])

# Créer et entraîner le modèle PPO
model = PPO('MlpPolicy', vec_env, verbose=1)
model.learn(total_timesteps=100000)

# Sauvegarder le modèle
model.save("ppo_forex_trading")

# Charger et utiliser le modèle
model = PPO.load("ppo_forex_trading")
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if dones[0]:
        obs = vec_env.reset()
```

### Méthode 3 : Utilisation avec RL Baselines3 Zoo

Le projet utilise un fork personnalisé de **RL Baselines3 Zoo** pour l'entraînement. Si le sous-module est configuré :

```bash
# Entraîner un modèle avec des hyperparamètres
python -m rl_zoo3.train \
    --algo ppo \
    --env ForexFullEURUSD-v1 \
    --hyperparams-file hyperparams/tuned/ForexFullEURUSD-v6/ppo.yml \
    --tensorboard-log logs/

# Évaluer un modèle entraîné
python -m rl_zoo3.enjoy \
    --algo ppo \
    --env ForexFullEURUSD-v1 \
    --folder logs/
```

## 🎯 Concepts clés

### Actions disponibles
- **BUY** : Ouvrir une position longue (acheter)
- **SELL** : Ouvrir une position courte (vendre)
- **CLOSE** : Fermer la position actuelle

### Environnements
- **ForexEnvBasic** : Environnement de base avec 3 actions
- Différentes variantes selon les actions autorisées

### Stratégies de récompense
- **ForexRewardStrategyLogPortfolioReturn** : Récompense basée sur le log du retour du portefeuille
- **ForexRewardStrategyWeightedLogPortfolioReturns** : Version pondérée

### Coûts de trading
- **ForexTradingCostsStrategySpread** : Coûts basés sur le spread bid-ask
- **ForexTradingCostsStrategyRelativeFee** : Coûts basés sur un pourcentage

### Algorithmes RL supportés
- **PPO** (Proximal Policy Optimization) - Recommandé
- **DQN** (Deep Q-Network)
- **A2C** (Advantage Actor-Critic)

## 📈 Résultats attendus

Selon le README, le modèle PPO optimisé a obtenu :
- **112.53%** de retour cumulatif sur la période de validation
- **46.31%** de retour cumulatif sur la période d'évaluation
- Ratio de Sharpe de **3.26** (validation) et **1.49** (évaluation)

⚠️ **Note importante** : Les résultats sont obtenus dans un environnement **sans commission**. L'ajout de commissions réduit significativement la rentabilité.

## 🔧 Configuration

### Modifier les chemins dans `config.py`
```python
DATA_PATH = "/chemin/vers/vos/donnees"
LOGS_PATH = "/chemin/vers/logs"
HYPERPARAMS_PATH = "/chemin/vers/hyperparams"
```

### Hyperparamètres
Les hyperparamètres sont stockés dans des fichiers YAML :
- `hyperparams/default/` : Valeurs par défaut
- `hyperparams/tuned/` : Valeurs optimisées

## 📝 Workflow typique

1. **Préparer les données** : Utiliser `notebooks/forex_data_collection.ipynb` et `forex_data_preproc_eda.ipynb`
2. **Créer des features** : Utiliser `forex_data_feature_engineering_basic.ipynb` ou `forex_data_feature_engineering_ta.ipynb`
3. **Entraîner un modèle** : Utiliser `forex_full_eurusd_rl_experiments.ipynb`
4. **Analyser les résultats** : Utiliser `forex_full_eurusd_best_rl_models_analysis.ipynb`

## ⚠️ Avertissements

1. **Données requises** : Le bot nécessite des données Forex historiques. Assurez-vous d'avoir les données dans le chemin spécifié dans `config.py`
2. **Temps d'entraînement** : L'entraînement peut prendre plusieurs heures selon la taille des données
3. **Risques** : Ce bot est à des fins éducatives/recherche. Ne l'utilisez pas avec de l'argent réel sans tests approfondis
4. **Commissions** : Les résultats sont meilleurs sans commissions. En conditions réelles avec commissions, les performances peuvent être très différentes

## 🆘 Dépannage

### Problème : Module non trouvé
```bash
# Assurez-vous d'être dans le répertoire du projet
cd /media/gryphen/Disque\ local/SERIE/LINUX/RL-trading-main

# Vérifiez que les chemins Python incluent le projet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Problème : Données manquantes
Vérifiez que les données Forex sont présentes dans le chemin spécifié dans `config.py`

### Problème : Erreur avec gym
```bash
pip install setuptools==65.5.0 pip==21 wheel==0.38.0
pip install gym==0.21.0
```

## 📚 Ressources supplémentaires

- Documentation Stable-Baselines3 : https://stable-baselines3.readthedocs.io/
- Documentation Gym : https://gymnasium.farama.org/
- Paper de référence : "Financial Trading as a Game: A Deep Reinforcement Learning Approach"

---

**Bon trading ! 🚀**

