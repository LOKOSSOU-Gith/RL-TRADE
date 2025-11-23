"""
Source de données Twelve Data pour Forex
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from twelve_data_config import TWELVE_DATA_API_KEY, FOREX_PAIR, TIMEFRAME, OUTPUT_SIZE, TIMEZONE

# Base URL de l'API Twelve Data
BASE_URL = "https://api.twelvedata.com"

# Mapping des timeframes vers les formats Twelve Data
TIMEFRAME_MAPPING = {
    "1min": "1min",
    "5min": "5min", 
    "15min": "15min",
    "30min": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day"
}

def get_historical_data(
    symbol: str = None,
    interval: str = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    outputsize: int = None
) -> pd.DataFrame:
    """
    Récupère les données historiques depuis Twelve Data API via HTTP.
    
    Args:
        symbol: Paire Forex (ex: 'EUR/USD' ou 'EURUSD'). Si None, utilise la config par défaut.
        interval: Timeframe (ex: '1min', '5min', '1h'). Si None, utilise la config par défaut.
        start_date: Date de début (format: 'YYYY-MM-DD'). Si None, utilise les dernières données.
        end_date: Date de fin (format: 'YYYY-MM-DD'). Si None, utilise les données jusqu'à présent.
        outputsize: Nombre de bougies à récupérer. Si None, utilise la config par défaut.
    
    Returns:
        DataFrame avec colonnes: ['datetime', 'open', 'high', 'low', 'close']
    """
    # Utiliser les valeurs par défaut de la configuration si non spécifiées
    symbol = symbol or FOREX_PAIR
    interval = interval or TIMEFRAME
    outputsize = outputsize or OUTPUT_SIZE
    
    # S'assurer que le symbole a le format correct (EUR/USD)
    if '/' not in symbol:
        # Ajouter le slash si manquant (EURUSD -> EUR/USD)
        if len(symbol) == 6:  # Format comme EURUSD
            symbol = symbol[:3] + '/' + symbol[3:]
    
    # Mapper le timeframe
    td_interval = TIMEFRAME_MAPPING.get(interval, interval)
    
    try:
        # Construire l'URL de la requête
        url = f"{BASE_URL}/time_series"
        
        # Paramètres de la requête
        params = {
            "symbol": symbol,
            "interval": td_interval,
            "outputsize": outputsize,
            "apikey": TWELVE_DATA_API_KEY
        }
        
        # Ajouter les dates si spécifiées
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
            
        # Faire la requête HTTP
        print(f"Récupération des données historiques pour {symbol} ({interval})...")
        response = requests.get(url, params=params)
        response.raise_for_status()  # Lève une exception si la requête échoue
        
        # Parser la réponse JSON
        data_json = response.json()
        
        # Vérifier s'il y a une erreur
        if "status_code" in data_json and data_json.get("status_code") == "error":
            raise ValueError(f"Erreur API: {data_json.get('message', 'Erreur inconnue')}")
        
        # Extraire les valeurs
        values = data_json.get("values", [])
        if not values:
            raise ValueError(f"Aucune donnée récupérée pour {symbol}")
            
        # Créer le DataFrame
        data = pd.DataFrame(values)
        
        # Renommer les colonnes pour correspondre au format FOREX_COLS
        column_mapping = {
            'datetime': '<DT>',
            'open': '<OPEN>', 
            'high': '<HIGH>',
            'low': '<LOW>',
            'close': '<CLOSE>'
        }
        
        data = data.rename(columns=column_mapping)
        
        # S'assurer que toutes les colonnes requises sont présentes
        required_cols = ['<DT>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Colonne manquante: {col}")
        
        # Convertir les types
        data['<DT>'] = pd.to_datetime(data['<DT>'])
        for col in ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']:
            data[col] = data[col].astype(np.float32)
            
        # Trier par date
        data = data.sort_values('<DT>').reset_index(drop=True)
        
        print(f"Données récupérées: {len(data)} bougies de {data['<DT>'].min()} à {data['<DT>'].max()}")
        
        return data
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données historiques: {e}")
        raise


def get_live_data(symbol: str = None, interval: str = None) -> pd.DataFrame:
    """
    Récupère la dernière bougie en temps réel depuis Twelve Data API via HTTP.
    
    Args:
        symbol: Paire Forex (ex: 'EUR/USD' ou 'EURUSD'). Si None, utilise la config par défaut.
        interval: Timeframe (ex: '1min', '5min', '1h'). Si None, utilise la config par défaut.
    
    Returns:
        DataFrame avec une seule ligne contenant la dernière bougie
    """
    # Utiliser les valeurs par défaut de la configuration si non spécifiées
    symbol = symbol or FOREX_PAIR
    interval = interval or TIMEFRAME
    
    # S'assurer que le symbole a le format correct (EUR/USD)
    if '/' not in symbol:
        # Ajouter le slash si manquant (EURUSD -> EUR/USD)
        if len(symbol) == 6:  # Format comme EURUSD
            symbol = symbol[:3] + '/' + symbol[3:]
    
    # Mapper le timeframe
    td_interval = TIMEFRAME_MAPPING.get(interval, interval)
    
    try:
        # Construire l'URL de la requête
        url = f"{BASE_URL}/time_series"
        
        # Paramètres de la requête (outputsize=1 pour la dernière bougie)
        params = {
            "symbol": symbol,
            "interval": td_interval,
            "outputsize": 1,
            "apikey": TWELVE_DATA_API_KEY
        }
            
        # Faire la requête HTTP
        print(f"Récupération des données live pour {symbol} ({interval})...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parser la réponse JSON
        data_json = response.json()
        
        # Vérifier s'il y a une erreur
        if "status_code" in data_json and data_json.get("status_code") == "error":
            raise ValueError(f"Erreur API: {data_json.get('message', 'Erreur inconnue')}")
        
        # Extraire les valeurs
        values = data_json.get("values", [])
        if not values:
            raise ValueError(f"Aucune donnée live récupérée pour {symbol}")
            
        # Créer le DataFrame
        data = pd.DataFrame(values)
        
        # Renommer les colonnes pour correspondre au format FOREX_COLS
        column_mapping = {
            'datetime': '<DT>',
            'open': '<OPEN>', 
            'high': '<HIGH>',
            'low': '<LOW>',
            'close': '<CLOSE>'
        }
        
        data = data.rename(columns=column_mapping)
        
        # S'assurer que toutes les colonnes requises sont présentes
        required_cols = ['<DT>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Colonne manquante: {col}")
        
        # Convertir les types
        data['<DT>'] = pd.to_datetime(data['<DT>'])
        for col in ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']:
            data[col] = data[col].astype(np.float32)
            
        print(f"Donnée live récupérée: {data['<DT>'].iloc[0]} - O:{data['<OPEN>'].iloc[0]} H:{data['<HIGH>'].iloc[0]} L:{data['<LOW>'].iloc[0]} C:{data['<CLOSE>'].iloc[0]}")
        
        return data
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données live: {e}")
        raise


def get_real_time_price(symbol: str = None) -> Dict[str, Any]:
    """
    Récupère le prix en temps réel (quote) pour une paire Forex via HTTP.
    
    Args:
        symbol: Paire Forex (ex: 'EUR/USD' ou 'EURUSD'). Si None, utilise la config par défaut.
    
    Returns:
        Dictionnaire avec les informations de prix
    """
    # Utiliser les valeurs par défaut de la configuration si non spécifiées
    symbol = symbol or FOREX_PAIR
    
    # S'assurer que le symbole a le format correct (EUR/USD)
    if '/' not in symbol:
        # Ajouter le slash si manquant (EURUSD -> EUR/USD)
        if len(symbol) == 6:  # Format comme EURUSD
            symbol = symbol[:3] + '/' + symbol[3:]
    
    try:
        # Construire l'URL de la requête
        url = f"{BASE_URL}/quote"
        
        # Paramètres de la requête
        params = {
            "symbol": symbol,
            "apikey": TWELVE_DATA_API_KEY
        }
            
        # Faire la requête HTTP
        print(f"Récupération du prix réel pour {symbol}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parser la réponse JSON
        data_json = response.json()
        
        # Vérifier s'il y a une erreur
        if "status_code" in data_json and data_json.get("status_code") == "error":
            raise ValueError(f"Erreur API: {data_json.get('message', 'Erreur inconnue')}")
        
        # Extraire les informations
        price_info = {
            'symbol': symbol,
            'price': float(data_json.get('price', 0)),
            'timestamp': pd.to_datetime(data_json.get('timestamp', datetime.now()))
        }
        
        print(f"Prix réel pour {symbol}: {price_info['price']}")
        
        return price_info
        
    except Exception as e:
        print(f"Erreur lors de la récupération du prix réel: {e}")
        raise


# Fonction utilitaire pour convertir au format attendu par le bot
def convert_to_forex_format(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les données Twelve Data au format attendu par le bot de trading.
    
    Args:
        data: DataFrame des données Twelve Data
        
    Returns:
        DataFrame formaté pour le bot
    """
    # Le format est déjà correct grâce aux fonctions ci-dessus
    # Cette fonction est une sécurité supplémentaire
    return data.copy()


# Test des fonctions
if __name__ == "__main__":
    # Test de récupération des données historiques
    try:
        hist_data = get_historical_data(outputsize=100)
        print(f"Test historique: {len(hist_data)} lignes récupérées")
        print(hist_data.head())
        print(hist_data.tail())
    except Exception as e:
        print(f"Erreur test historique: {e}")
    
    # Test de récupération des données live
    try:
        live_data = get_live_data()
        print(f"Test live: {len(live_data)} ligne récupérée")
        print(live_data)
    except Exception as e:
        print(f"Erreur test live: {e}")
