import asyncio
import json
import os
import pickle
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional, Any
from json_utils import dumps 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("behavioral_baseline")

# Database connection parameters (can be overridden by environment variables)
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = int(os.environ.get('DB_PORT'))
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_NAME = os.environ.get('DB_NAME')

# Interval for updating baselines (in seconds)
BASELINE_UPDATE_INTERVAL = int(os.getenv('BASELINE_UPDATE_INTERVAL', 300))

# Model storage
MODEL_DIR = os.environ.get('MODEL_DIR', './models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Minimum data points required for training
MIN_DATA_POINTS = int(os.getenv('MIN_DATA_POINTS', 24))  # At least 48 data points for meaningful training

# Global variables to store models
prophet_models = {}
isolation_forests = {}
scalers = {}

class DatabaseConnection:
    """Context manager for database connections"""
    def __enter__(self):
        self.conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

def create_tables():
    """Create necessary tables if they don't exist"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Table for storing baseline information
            cur.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    id SERIAL PRIMARY KEY,
                    application VARCHAR(100),
                    metric_name VARCHAR(50),
                    model_type VARCHAR(20),
                    model_path VARCHAR(255),
                    last_updated TIMESTAMP,
                    last_training_data TIMESTAMP,
                    accuracy_score FLOAT,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(application, metric_name, model_type)
                )
            """)
            
            # Table for storing anomaly detections
            cur.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    id SERIAL PRIMARY KEY,
                    application VARCHAR(100),
                    metric_name VARCHAR(50),
                    detection_time TIMESTAMP,
                    value FLOAT,
                    expected_range_lower FLOAT,
                    expected_range_upper FLOAT,
                    anomaly_score FLOAT,
                    severity VARCHAR(20),
                    details JSONB,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.commit()
            logger.info("Database tables created or verified successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def fetch_historical_data(application: str, metric_name: str, days: int = 7) -> List[Tuple]:
    """Fetch historical data for a specific application and metric"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Map metric names to database columns
            metric_column_map = {
                'cpu_usage': 'cpu_usage',
                'memory_usage': 'memory_usage',
                'disk_read_iops': 'disk_read_iops',
                'disk_write_iops': 'disk_write_iops',
                'network_packets_sent_per_sec': 'network_packets_sent_per_sec',
                'network_packets_recv_per_sec': 'network_packets_recv_per_sec',
                'gpu_usage': 'gpu_usage',
                'gpu_memory_usage': 'gpu_memory_usage'
            }
            
            if metric_name not in metric_column_map:
                logger.error(f"Unknown metric name: {metric_name}")
                return []
            
            column = metric_column_map[metric_name]
            
            # Query to get historical data
            cur.execute(f"""
                SELECT time, {column} 
                FROM metrics 
                WHERE application = %s AND {column} IS NOT NULL
                AND time >= NOW() - INTERVAL '{days} days'
                ORDER BY time
            """, (application,))
            
            rows = cur.fetchall()
            return rows
    except Exception as e:
        logger.error(f"Error fetching historical data for {application}/{metric_name}: {e}")
        return []

def load_model(application: str, metric_name: str, model_type: str) -> Optional[Any]:
    """Load a trained model from disk"""
    model_path = os.path.join(MODEL_DIR, f"{application}_{metric_name}_{model_type}.pkl")
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.PickleError) as e:
        logger.warning(f"Could not load model {model_path}: {e}")
        return None

def save_model(application: str, metric_name: str, model_type: str, model: Any) -> bool:
    """Save a trained model to disk"""
    model_path = os.path.join(MODEL_DIR, f"{application}_{metric_name}_{model_type}.pkl")
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return True
    except (IOError, pickle.PickleError) as e:
        logger.error(f"Error saving model {model_path}: {e}")
        return False

def train_prophet_model(application: str, metric_name: str, historical_data: List[Tuple]) -> bool:
    """Train Prophet model for an application and metric"""
    try:
        if len(historical_data) < MIN_DATA_POINTS:
            logger.warning(f"Insufficient data for Prophet model training for {application}/{metric_name}: {len(historical_data)} points")
            return False
            
        df = pd.DataFrame(historical_data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Handle missing values
        df = df.dropna()
        
        if len(df) < MIN_DATA_POINTS:
            logger.warning(f"Insufficient valid data for Prophet model training for {application}/{metric_name}: {len(df)} points")
            return False
        
        # Train the model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # Disable yearly seasonality unless we have enough data
            changepoint_prior_scale=0.05
        )
        model.fit(df)
        
        # Save the model
        if save_model(application, metric_name, "prophet", model):
            prophet_models[f"{application}_{metric_name}"] = model
            
            # Update the database
            with DatabaseConnection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO baselines (application, metric_name, model_type, model_path, last_updated, last_training_data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (application, metric_name, model_type) 
                    DO UPDATE SET 
                        model_path = EXCLUDED.model_path,
                        last_updated = EXCLUDED.last_updated,
                        last_training_data = EXCLUDED.last_training_data
                """, (
                    application, 
                    metric_name, 
                    "prophet", 
                    f"{MODEL_DIR}/{application}_{metric_name}_prophet.pkl",
                    datetime.now(),
                    df['ds'].max()
                ))
            
            logger.info(f"Prophet model trained for {application}/{metric_name}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error training Prophet model for {application}/{metric_name}: {e}")
        return False

def train_isolation_forest(application: str, metric_name: str, historical_data: List[Tuple]) -> bool:
    """Train Isolation Forest model for an application and metric"""
    try:
        if len(historical_data) < MIN_DATA_POINTS:
            logger.warning(f"Insufficient data for Isolation Forest training for {application}/{metric_name}: {len(historical_data)} points")
            return False
            
        df = pd.DataFrame(historical_data, columns=['ds', 'y'])
        df = df.dropna()
        
        if len(df) < MIN_DATA_POINTS:
            logger.warning(f"Insufficient valid data for Isolation Forest training for {application}/{metric_name}: {len(df)} points")
            return False
        
        # Extract features (we'll use the value and some derived features)
        features = df[['y']].copy()
        
        # Add time-based features
        features['hour'] = df['ds'].dt.hour
        features['day_of_week'] = df['ds'].dt.dayofweek
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train the model
        model = IsolationForest(
            contamination=0.10,  # Expected proportion of anomalies
            random_state=42,
            n_estimators=100
        )
        model.fit(scaled_features)
        
        # Save the model and scaler
        model_key = f"{application}_{metric_name}"
        if save_model(application, metric_name, "isolation_forest", model) and \
           save_model(application, metric_name, "scaler", scaler):
            isolation_forests[model_key] = model
            scalers[model_key] = scaler
            
            # Update the database
            with DatabaseConnection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO baselines (application, metric_name, model_type, model_path, last_updated, last_training_data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (application, metric_name, model_type) 
                    DO UPDATE SET 
                        model_path = EXCLUDED.model_path,
                        last_updated = EXCLUDED.last_updated,
                        last_training_data = EXCLUDED.last_training_data
                """, (
                    application, 
                    metric_name, 
                    "isolation_forest", 
                    f"{MODEL_DIR}/{application}_{metric_name}_isolation_forest.pkl",
                    datetime.now(),
                    df['ds'].max()
                ))
            
            logger.info(f"Isolation Forest model trained for {application}/{metric_name}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error training Isolation Forest model for {application}/{metric_name}: {e}")
        return False

def load_existing_models():
    """Load all existing models from disk"""
    global prophet_models, isolation_forests, scalers
    
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT application, metric_name, model_type FROM baselines WHERE is_active = TRUE")
            rows = cur.fetchall()
            
            for application, metric_name, model_type in rows:
                if model_type == "prophet":
                    model = load_model(application, metric_name, "prophet")
                    if model:
                        prophet_models[f"{application}_{metric_name}"] = model
                        logger.info(f"Loaded Prophet model for {application}/{metric_name}")
                
                elif model_type == "isolation_forest":
                    model = load_model(application, metric_name, "isolation_forest")
                    scaler = load_model(application, metric_name, "scaler")
                    if model and scaler:
                        model_key = f"{application}_{metric_name}"
                        isolation_forests[model_key] = model
                        scalers[model_key] = scaler
                        logger.info(f"Loaded Isolation Forest model for {application}/{metric_name}")
    except Exception as e:
        logger.error(f"Error loading existing models: {e}")

async def update_baselines():
    """Update baselines for all applications and metrics"""
    while True:
        try:
            logger.info("Starting baseline update process")
            
            # Get all applications
            with DatabaseConnection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT application FROM metrics")
                applications = [row[0] for row in cur.fetchall()]
            
            # Metrics to track for each application
            metrics_to_track = [
                'cpu_usage', 'memory_usage', 'disk_read_iops', 'disk_write_iops',
                'network_packets_sent_per_sec', 'network_packets_recv_per_sec'
            ]
            
            # Add GPU metrics if available
            with DatabaseConnection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT application FROM metrics WHERE gpu_usage IS NOT NULL")
                gpu_applications = [row[0] for row in cur.fetchall()]
                
                if gpu_applications:
                    metrics_to_track.extend(['gpu_usage', 'gpu_memory_usage'])
            
            # Train models for each application and metric
            for application in applications:
                for metric_name in metrics_to_track:
                    # Skip GPU metrics for applications that don't have them
                    if metric_name.startswith('gpu') and application not in gpu_applications:
                        continue
                    
                    # Fetch historical data
                    historical_data = fetch_historical_data(application, metric_name)
                    
                    if historical_data:
                        # Train Prophet model
                        train_prophet_model(application, metric_name, historical_data)
                        
                        # Train Isolation Forest model
                        train_isolation_forest(application, metric_name, historical_data)
            
            logger.info("Baseline update process completed")
            await asyncio.sleep(BASELINE_UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Error in baseline update process: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying

def detect_anomalies(application: str, metric_name: str, new_data: List[Tuple]) -> List[Dict]:
    """Detect anomalies using Isolation Forest"""
    model_key = f"{application}_{metric_name}"
    
    if model_key not in isolation_forests:
        logger.warning(f"No Isolation Forest model found for {application}/{metric_name}")
        return []
    
    try:
        model = isolation_forests[model_key]
        scaler = scalers[model_key]
        
        # Prepare the data
        df = pd.DataFrame(new_data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.dropna()
        
        if df.empty:
            return []
        
        # Extract features
        features = df[['y']].copy()
        features['hour'] = df['ds'].dt.hour
        features['day_of_week'] = df['ds'].dt.dayofweek
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Predict anomalies
        predictions = model.predict(scaled_features)
        anomaly_scores = model.decision_function(scaled_features)
        
        # Identify anomalies
        anomalies = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if predictions[i] == -1:  # Anomaly detected
                anomalies.append({
                    'timestamp': row['ds'],
                    'value': row['y'],
                    'anomaly_score': float(anomaly_scores[i]),
                    'severity': 'high' if anomaly_scores[i] < -0.5 else 'medium'
                })
        
        # Store anomalies in database
        if anomalies:
            with DatabaseConnection() as conn:
                cur = conn.cursor()
                for anomaly in anomalies:
                    cur.execute("""
                        INSERT INTO anomaly_detections 
                        (application, metric_name, detection_time, value, anomaly_score, severity, details)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        application,
                        metric_name,
                        anomaly['timestamp'],
                        anomaly['value'],
                        anomaly['anomaly_score'],
                        anomaly['severity'],
                        dumps(anomaly)
                    ))
        
        return anomalies
    except Exception as e:
        logger.error(f"Error detecting anomalies for {application}/{metric_name}: {e}")
        return []

def make_forecasts(application: str, metric_name: str, periods: int = 24) -> List[Dict]:
    """Make forecasts using Prophet"""
    model_key = f"{application}_{metric_name}"
    
    if model_key not in prophet_models:
        logger.warning(f"No Prophet model found for {application}/{metric_name}")
        return []
    
    try:
        model = prophet_models[model_key]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='H')
        forecast = model.predict(future)
        
        # Extract relevant columns
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        # Convert to list of dictionaries
        forecasts = []
        for _, row in result.iterrows():
            forecasts.append({
                'timestamp': row['ds'],
                'predicted_value': row['yhat'],
                'lower_bound': row['yhat_lower'],
                'upper_bound': row['yhat_upper']
            })
        
        return forecasts
    except Exception as e:
        logger.error(f"Error making forecasts for {application}/{metric_name}: {e}")
        return []

def get_current_baseline(application: str, metric_name: str) -> Optional[Dict]:
    """Get the current baseline for an application and metric"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT model_type, last_updated, last_training_data 
                FROM baselines 
                WHERE application = %s AND metric_name = %s AND is_active = TRUE
            """, (application, metric_name))
            
            rows = cur.fetchall()
            if not rows:
                return None
            
            result = {}
            for model_type, last_updated, last_training_data in rows:
                result[model_type] = {
                    'last_updated': last_updated,
                    'last_training_data': last_training_data
                }
            
            return result
    except Exception as e:
        logger.error(f"Error getting baseline for {application}/{metric_name}: {e}")
        return None

async def main():
    """Main function to run the service"""
    logger.info("Starting Behavioral Baseline Service")
    
    # Create database tables if they don't exist
    create_tables()
    
    # Load existing models
    load_existing_models()
    
    # Start the baseline update task
    await update_baselines()

if __name__ == "__main__":
    asyncio.run(main())