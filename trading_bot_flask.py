"""
Bot de trading avec interface Flask - Cycle Apprentissage/Trading optimisé
Limité à 800 requêtes/jour avec Twelve Data API
"""

import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

# Import des modules du bot
from rl_trading.data.forex.twelve_data_source import get_historical_data, get_live_data
from rl_trading.environments import ForexEnvBasic, Actions, ForexMarketOrderStrategyAllIn, ForexRewardStrategyLogPortfolioReturn, ForexTradingCostsStrategySpread
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Configuration Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_bot_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Variables globales pour le bot
bot_state = {
    'model': None,
    'environment': None,
    'current_position': 'CLOSE',
    'portfolio_value': 1000.0,
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'last_price': 0.0,
    'trade_history': deque(maxlen=50),
    'price_history': deque(maxlen=100),
    'is_running': False,
    'current_phase': 'IDLE',  # IDLE, TRAINING, TRADING
    'cycle_count': 0,
    'api_requests_today': 0,
    'api_limit': 800,
    'last_prediction': 'CLOSE',
    'last_reward': 0.0,
    'total_reward': 0.0,
    'training_progress': 0,
    'cycle_performance': [],
    'session_start': datetime.now()
}

class OptimizedTradingBot:
    def __init__(self):
        self.model = None
        self.env = None
        self.data_buffer = deque(maxlen=10000)  # Buffer plus grand
        self.last_update = datetime.now()
        self.cycle_start_time = None
        
    def check_api_limit(self):
        """Vérifier si on peut faire des requêtes API"""
        if bot_state['api_requests_today'] >= bot_state['api_limit']:
            return False
        return True
    
    def track_api_usage(self, requests_used=1):
        """Suivre l'utilisation de l'API"""
        bot_state['api_requests_today'] += requests_used
        
        # Réinitialiser tous les jours à minuit
        if datetime.now().date() > bot_state['session_start'].date():
            bot_state['api_requests_today'] = requests_used
            bot_state['session_start'] = datetime.now()
    
    def initialize_with_api_limit(self):
        """Initialiser le bot en respectant la limite API"""
        try:
            if not self.check_api_limit():
                print("❌ Limite API atteinte pour aujourd'hui")
                return False
            
            print("🚀 Initialisation optimisée du bot...")
            bot_state['current_phase'] = 'INITIALIZING'
            
            # Récupérer les données historiques (une grosse requête)
            print("📊 Récupération des données historiques...")
            historical_data = get_historical_data(
                symbol='EUR/USD', 
                interval='15min',  # Interval plus grand pour économiser des requêtes
                outputsize=3000     # Moins de données mais suffisantes
            )
            self.track_api_usage(1)
            
            if len(historical_data) < 1000:
                print("❌ Pas assez de données historiques")
                return False
            
            # Stocker les données
            for _, row in historical_data.iterrows():
                self.data_buffer.append(row.to_dict())
            
            # Créer l'environnement
            print("🤖 Création de l'environnement...")
            self.env = ForexEnvBasic(
                target_prices_df=historical_data,
                features_df=historical_data.drop('<DT>', axis=1),
                portfolio_value=1000,
                allowed_actions={Actions.BUY, Actions.SELL, Actions.CLOSE},
                market_order_strategy=ForexMarketOrderStrategyAllIn(),
                reward_strategy=ForexRewardStrategyLogPortfolioReturn(),
                trading_costs_strategy=ForexTradingCostsStrategySpread(spread=0.0001),
                include_in_obs=['position']
            )
            
            # Créer et entraîner le modèle initial
            print("🎓 Création du modèle PPO...")
            vec_env = DummyVecEnv([lambda: self.env])
            
            # Paramètres PPO optimisés pour éviter l'erreur d'index
            self.model = PPO(
                'MlpPolicy', 
                vec_env, 
                verbose=0, 
                learning_rate=0.002,
                n_steps=1024,  # Réduit de 2048 à 1024 pour éviter l'erreur
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2
            )
            
            # Mettre à jour l'état
            bot_state['model'] = self.model
            bot_state['environment'] = self.env
            bot_state['last_price'] = float(historical_data['<CLOSE>'].iloc[-1])
            
            print(f"✅ Bot initialisé! Requêtes API utilisées: {bot_state['api_requests_today']}/{bot_state['api_limit']}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur d'initialisation: {e}")
            return False
    
    def training_phase(self, duration_minutes=5):
        """Phase d'entraînement de quelques minutes"""
        try:
            if not self.check_api_limit():
                print("❌ Limite API atteinte - pause de l'entraînement")
                return False
            
            print(f"🎓 Phase d'entraînement ({duration_minutes} minutes)...")
            bot_state['current_phase'] = 'TRAINING'
            self.cycle_start_time = datetime.now()
            
            # Utiliser les données du buffer pour l'entraînement
            if len(self.data_buffer) < 2000:
                print("❌ Pas assez de données pour l'entraînement")
                return False
            
            training_data = pd.DataFrame(list(self.data_buffer)[-2000:])
            
            # Recréer l'environnement avec les données récentes
            self.env = ForexEnvBasic(
                target_prices_df=training_data,
                features_df=training_data.drop('<DT>', axis=1),
                portfolio_value=bot_state['portfolio_value'],
                allowed_actions={Actions.BUY, Actions.SELL, Actions.CLOSE},
                market_order_strategy=ForexMarketOrderStrategyAllIn(),
                reward_strategy=ForexRewardStrategyLogPortfolioReturn(),
                trading_costs_strategy=ForexTradingCostsStrategySpread(spread=0.0001),
                include_in_obs=['position']
            )
            
            # Entraîner le modèle
            vec_env = DummyVecEnv([lambda: self.env])
            self.model.set_env(vec_env)
            
            # Calculer le nombre de timesteps pour la durée spécifiée
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            total_timesteps = 0
            
            while datetime.now() < end_time and bot_state['is_running']:
                # Entraîner par petits lots avec n_steps réduit
                self.model.learn(total_timesteps=256, reset_num_timesteps=False)  # Réduit à 256
                total_timesteps += 256
                
                bot_state['training_progress'] += 1
                
                # Envoyer la progression
                socketio.emit('training_update', {
                    'progress': bot_state['training_progress'],
                    'timesteps': total_timesteps,
                    'phase': 'TRAINING',
                    'remaining_time': str(end_time - datetime.now()).split('.')[0]
                })
                
                print(f"📚 Entraînement: {total_timesteps} timesteps - Temps restant: {end_time - datetime.now()}")
                time.sleep(2)  # Petite pause pour éviter la surcharge
            
            print(f"✅ Entraînement terminé: {total_timesteps} timesteps")
            return True
            
        except Exception as e:
            print(f"❌ Erreur durant l'entraînement: {e}")
            return False
    
    def trading_phase(self, duration_minutes=3):
        """Phase de trading de quelques minutes"""
        try:
            if not self.check_api_limit():
                print("❌ Limite API atteinte - pause du trading")
                return False
            
            print(f"💼 Phase de trading ({duration_minutes} minutes)...")
            bot_state['current_phase'] = 'TRADING'
            
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            trades_made = 0
            phase_reward = 0
            
            while datetime.now() < end_time and bot_state['is_running']:
                try:
                    # Récupérer les données live (1 requête)
                    live_data = get_live_data(symbol='EUR/USD', interval='1min')
                    self.track_api_usage(1)
                    
                    if not live_data.empty:
                        current_price = float(live_data['<CLOSE>'].iloc[0])
                        current_data = live_data.iloc[0]
                        
                        # Ajouter au buffer
                        self.data_buffer.append(current_data.to_dict())
                        
                        # Prédire l'action
                        action, confidence = self.predict_action(live_data)
                        bot_state['last_prediction'] = action
                        
                        # Exécuter le trade
                        trade_result = self.execute_trade(action, current_price)
                        if trade_result:
                            trades_made += 1
                            phase_reward += bot_state['last_reward']
                        
                        # Envoyer les mises à jour
                        socketio.emit('trading_update', {
                            'action': action,
                            'price': current_price,
                            'confidence': confidence,
                            'trades_made': trades_made,
                            'phase_reward': phase_reward,
                            'remaining_time': str(end_time - datetime.now()).split('.')[0],
                            'api_usage': f"{bot_state['api_requests_today']}/{bot_state['api_limit']}"
                        })
                        
                        print(f"💹 Trade: {action} @ {current_price} (Confiance: {confidence:.2f}) - Temps restant: {end_time - datetime.now()}")
                    
                    # Attendre entre les trades pour économiser l'API
                    time.sleep(30)  # 30 secondes entre chaque requête
                    
                except Exception as e:
                    print(f"❌ Erreur dans la boucle de trading: {e}")
                    time.sleep(60)
            
            # Sauvegarder la performance du cycle
            cycle_performance = {
                'cycle': bot_state['cycle_count'],
                'phase': 'TRADING',
                'trades_made': trades_made,
                'reward': phase_reward,
                'duration': str(datetime.now() - (end_time - timedelta(minutes=duration_minutes))).split('.')[0]
            }
            bot_state['cycle_performance'].append(cycle_performance)
            
            print(f"✅ Phase de trading terminée: {trades_made} trades, Récompense: {phase_reward:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur durant le trading: {e}")
            return False
    
    def predict_action(self, current_data):
        """Prédire l'action avec le modèle actuel"""
        try:
            if self.model is None or self.env is None:
                return 'CLOSE', 0.0
            
            # Préparer l'observation
            features = current_data.drop('<DT>', axis=1).values.reshape(1, -1)
            
            # Prédire l'action
            action, _states = self.model.predict(features, deterministic=True)
            
            # Convertir en texte
            action_map = {0: 'CLOSE', 1: 'BUY', 2: 'SELL'}
            predicted_action = action_map[action[0]]
            
            # Calculer la confiance
            try:
                action_probs = self.model.policy.action_probability(features)
                confidence = float(np.max(action_probs))
            except:
                confidence = 0.5
            
            return predicted_action, confidence
            
        except Exception as e:
            print(f"❌ Erreur de prédiction: {e}")
            return 'CLOSE', 0.0
    
    def execute_trade(self, action, current_price):
        """Exécuter un trade et mettre à jour l'état"""
        try:
            old_position = bot_state['current_position']
            
            # Simuler l'exécution
            if action == 'BUY' and old_position != 'BUY':
                bot_state['current_position'] = 'BUY'
                bot_state['total_trades'] += 1
            elif action == 'SELL' and old_position != 'SELL':
                bot_state['current_position'] = 'SELL'
                bot_state['total_trades'] += 1
            elif action == 'CLOSE' and old_position != 'CLOSE':
                bot_state['current_position'] = 'CLOSE'
                bot_state['total_trades'] += 1
            
            # Calculer la récompense
            price_change = (current_price - bot_state['last_price']) / bot_state['last_price']
            reward = price_change * 1000
            
            bot_state['last_reward'] = reward
            bot_state['total_reward'] += reward
            bot_state['last_price'] = current_price
            
            # Statistiques
            if reward > 0:
                bot_state['winning_trades'] += 1
            elif reward < 0:
                bot_state['losing_trades'] += 1
            
            # Ajouter à l'historique
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'price': current_price,
                'reward': reward,
                'portfolio_value': bot_state['portfolio_value']
            }
            bot_state['trade_history'].append(trade_record)
            bot_state['price_history'].append({
                'timestamp': datetime.now().isoformat(),
                'price': current_price
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur d'exécution du trade: {e}")
            return False
    
    def run_cycle(self):
        """Cycle complet: apprentissage + trading"""
        while bot_state['is_running']:
            try:
                if not self.check_api_limit():
                    print("⏳ Limite API atteinte - Attente du prochain jour...")
                    socketio.emit('api_limit_reached', {
                        'message': 'Limite API atteinte - Attente...',
                        'api_usage': f"{bot_state['api_requests_today']}/{bot_state['api_limit']}"
                    })
                    time.sleep(3600)  # Attendre 1 heure
                    continue
                
                bot_state['cycle_count'] += 1
                print(f"\n🔄 Cycle #{bot_state['cycle_count']} - Requêtes API: {bot_state['api_requests_today']}/{bot_state['api_limit']}")
                
                # Envoyer le début du cycle
                socketio.emit('cycle_start', {
                    'cycle': bot_state['cycle_count'],
                    'phase': 'TRAINING',
                    'api_usage': f"{bot_state['api_requests_today']}/{bot_state['api_limit']}"
                })
                
                # Phase 1: Entraînement (5 minutes)
                if self.training_phase(duration_minutes=5):
                    print("✅ Entraînement du cycle terminé")
                    socketio.emit('training_complete', {
                        'cycle': bot_state['cycle_count'],
                        'message': 'Entraînement terminé'
                    })
                else:
                    print("❌ Échec de l'entraînement")
                    socketio.emit('training_error', {
                        'cycle': bot_state['cycle_count'],
                        'message': 'Échec de l\'entraînement'
                    })
                    time.sleep(60)
                    continue
                
                # Petite pause entre les phases
                time.sleep(10)
                
                # Phase 2: Trading (3 minutes)
                if self.trading_phase(duration_minutes=3):
                    print("✅ Trading du cycle terminé")
                    socketio.emit('trading_complete', {
                        'cycle': bot_state['cycle_count'],
                        'message': 'Trading terminé'
                    })
                else:
                    print("❌ Échec du trading")
                    socketio.emit('trading_error', {
                        'cycle': bot_state['cycle_count'],
                        'message': 'Échec du trading'
                    })
                    time.sleep(60)
                    continue
                
                # Pause entre les cycles
                print("⏳ Pause de 2 minutes avant le prochain cycle...")
                bot_state['current_phase'] = 'RESTING'
                socketio.emit('cycle_rest', {
                    'cycle': bot_state['cycle_count'],
                    'message': 'Pause avant prochain cycle',
                    'duration': '2 minutes'
                })
                time.sleep(120)  # 2 minutes de pause
                
            except Exception as e:
                print(f"❌ Erreur dans le cycle: {e}")
                socketio.emit('cycle_error', {
                    'cycle': bot_state['cycle_count'],
                    'error': str(e)
                })
                time.sleep(300)  # 5 minutes de pause en cas d'erreur

# Instance globale du bot
trading_bot = OptimizedTradingBot()

# Routes Flask
@app.route('/')
def index():
    return render_template('trading_dashboard.html')

@app.route('/api/status')
def get_status():
    # Convertir les deque en listes pour la sérialisation JSON
    safe_state = {
        **bot_state,
        'trade_history': list(bot_state['trade_history']),
        'price_history': list(bot_state['price_history']),
        'cycle_performance': bot_state['cycle_performance']
    }
    return jsonify(safe_state)

@app.route('/api/trade_history')
def get_trade_history():
    return jsonify(list(bot_state['trade_history']))

@app.route('/api/price_history')
def get_price_history():
    return jsonify(list(bot_state['price_history']))

@app.route('/api/start', methods=['POST'])
def start_bot():
    try:
        # Initialiser le bot
        if not trading_bot.initialize_with_api_limit():
            return jsonify({'success': False, 'message': 'Échec de l\'initialisation'})
        
        # Démarrer les cycles
        bot_state['is_running'] = True
        
        cycle_thread = threading.Thread(target=trading_bot.run_cycle, daemon=True)
        cycle_thread.start()
        
        return jsonify({'success': True, 'message': f'Bot démarré - Cycle Apprentissage/Trading actif'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    bot_state['is_running'] = False
    bot_state['current_phase'] = 'IDLE'
    return jsonify({'success': True, 'message': 'Bot arrêté'})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print("🔌 Client connecté")
    # Convertir les deque en listes pour la sérialisation JSON
    safe_state = {
        **bot_state,
        'trade_history': list(bot_state['trade_history']),
        'price_history': list(bot_state['price_history']),
        'cycle_performance': bot_state['cycle_performance']
    }
    emit('status_update', safe_state)

@socketio.on('disconnect')
def handle_disconnect():
    print("🔌 Client déconnecté")

if __name__ == '__main__':
    print("🚀 Démarrage du bot optimisé - Cycle Apprentissage/Trading")
    print(f"📊 Limite API: {bot_state['api_limit']} requêtes/jour")
    print("⏱️  Cycle: 5min entraînement + 3min trading + 2min pause")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
