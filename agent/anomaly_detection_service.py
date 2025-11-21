import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from json_utils import dumps 

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# --- Configuration ---
# Use the same DB connection as your other services
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = int(os.environ.get('DB_PORT'))
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_NAME = os.environ.get('DB_NAME')

# --- Global Variables ---
# A list of critical applications that should be prioritized
CRITICAL_APPLICATIONS = os.getenv('CRITICAL_APPLICATIONS', 'web_server,api_gateway,database').split(',')

# A list of key metrics to check for correlation
KEY_METRICS_TO_CORRELATE = os.getenv('KEY_METRICS_TO_CORRELATE', 'cpu_usage,memory_usage,disk_read_iops,disk_write_iops,network_packets_sent_per_sec,network_packets_recv_per_sec').split(',')

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("anomaly_detection")

# --- Database Connection (Reused from other services) ---
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

# --- AI Model ---
# Load a pre-trained model for creating embeddings. This is done once at startup.
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Sentence Transformer model: {e}")
    embedding_model = None

# --- Core Service Functions ---

def create_tables():
    """Create necessary tables if they don't exist"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Table for storing the final, enriched alerts
            cur.execute("""
                CREATE TABLE IF NOT EXISTS enriched_alerts (
                    id SERIAL PRIMARY KEY,
                    source_anomaly_id INTEGER REFERENCES anomaly_detections(id),
                    application VARCHAR(100),
                    metric_name VARCHAR(50),
                    detection_time TIMESTAMP,
                    severity VARCHAR(20),
                    enriched_details JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Table to store embeddings of past anomalies for pattern matching
            cur.execute("""
                CREATE TABLE IF NOT EXISTS historical_anomaly_embeddings (
                    id SERIAL PRIMARY KEY,
                    anomaly_id INTEGER REFERENCES anomaly_detections(id),
                    embedding BYTEA -- Store as binary data instead of vector
                )
            """)
            
            # Create a regular B-tree index on anomaly_id for efficient lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_anomaly_embeddings_anomaly_id 
                ON historical_anomaly_embeddings (anomaly_id)
            """)
            
            conn.commit()
            logger.info("Anomaly Detection tables created or verified successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def fetch_unprocessed_anomalies(limit: int = 10) -> List[Dict]:
    """Fetch anomalies that haven't been enriched yet."""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, application, metric_name, detection_time, value, details 
                FROM anomaly_detections 
                WHERE acknowledged = FALSE 
                ORDER BY detection_time DESC 
                LIMIT %s
            """, (limit,))
            
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching unprocessed anomalies: {e}")
        return []

def get_correlated_metrics(anomaly: Dict) -> List[Dict]:
    """Check for other anomalous metrics at the same time for the same application."""
    correlated = []
    app = anomaly['application']
    detection_time = anomaly['detection_time']
    
    # Look for other anomalies within a 1-minute window
    time_window = timedelta(minutes=1)
    
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT metric_name, value 
                FROM anomaly_detections 
                WHERE application = %s 
                AND detection_time BETWEEN %s AND %s
                AND id != %s
            """, (app, detection_time - time_window, detection_time + time_window, anomaly['id']))
            
            columns = [desc[0] for desc in cur.description]
            correlated = [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching correlated metrics: {e}")
        
    return correlated

def get_contextual_data(anomaly: Dict) -> Dict:
    """
    Enrich the anomaly with context. 
    In a real system, this would query a CI/CD API, change management DB, etc.
    Here, we simulate it by checking for a local file.
    """
    context = {}
    app = anomaly['application']
    detection_time = anomaly['detection_time']
    
    # Simulate checking for a recent deployment
    log_file = 'simulated_deployment_log.json'
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                deployments = json.load(f)
            
            for deploy in deployments:
                if deploy['app'] == app:
                    deploy_time = datetime.fromisoformat(deploy['timestamp'])
                    # Check if deployment was within the last 10 minutes
                    if detection_time - deploy_time < timedelta(minutes=10):
                        context['recent_deployment'] = deploy
                        break
        except Exception as e:
            logger.warning(f"Could not parse deployment log: {e}")
            
    return context

def find_historical_matches(anomaly: Dict, correlated_metrics: List[Dict]) -> List[Dict]:
    """Find similar past anomalies using vector embeddings."""
    if not embedding_model:
        logger.warning("Embedding model not loaded. Skipping historical match.")
        return []

    # 1. Create a "fingerprint" text string of the current event
    fingerprint = f"Anomaly in {anomaly['application']}: {anomaly['metric_name']} is {anomaly['value']}."
    if correlated_metrics:
        fingerprint += f" Correlated with: {', '.join([m['metric_name'] for m in correlated_metrics])}."
    
    # 2. Generate the embedding for the current fingerprint
    current_embedding = embedding_model.encode(fingerprint)
    
    matches = []
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            # 3. Fetch all historical embeddings and calculate similarity manually
            cur.execute("""
                SELECT ha.anomaly_id, ad.details, ha.embedding
                FROM historical_anomaly_embeddings ha
                JOIN anomaly_detections ad ON ha.anomaly_id = ad.id
            """)
            
            rows = cur.fetchall()
            
            # Calculate cosine similarity manually
            for row in rows:
                historical_embedding = np.frombuffer(row[2], dtype=np.float32)
                similarity = np.dot(current_embedding, historical_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(historical_embedding)
                )
                
                if similarity > 0.8:  # Similarity threshold
                    matches.append({
                        'anomaly_id': row[0],
                        'details': row[1],
                        'similarity': similarity
                    })
            
            # Sort by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            matches = matches[:5]  # Limit to top 5 matches
            
    except Exception as e:
        logger.error(f"Error finding historical matches: {e}")

    return matches

def calculate_severity(anomaly: Dict, correlated_metrics: List[Dict], historical_matches: List[Dict]) -> str:
    """Calculate a severity score based on multiple factors."""
    score = 0
    
    # Base score from the original anomaly
    if anomaly.get('severity') == 'high':
        score += 30
    else:
        score += 10

    # Add score for each correlated metric
    score += len(correlated_metrics) * 15
    
    # Add score if historical matches exist
    if historical_matches:
        score += 20
        # Bonus if matches were severe
        if any('high' in match.get('details', {}).get('severity', '') for match in historical_matches):
            score += 15

    # Add score if the application is critical
    if anomaly['application'] in CRITICAL_APPLICATIONS:
        score += 25

    # Determine final severity level
    if score >= 70:
        return 'critical'
    elif score >= 50:
        return 'high'
    elif score >= 30:
        return 'medium'
    else:
        return 'low'

def store_and_index_enriched_alert(enriched_alert: Dict):
    """Save the final alert and create its embedding for future matching."""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # 1. Store the main enriched alert
            cur.execute("""
                INSERT INTO enriched_alerts (source_anomaly_id, application, metric_name, detection_time, severity, enriched_details)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                enriched_alert['source_anomaly_id'],
                enriched_alert['application'],
                enriched_alert['metric_name'],
                enriched_alert['detection_time'],
                enriched_alert['severity'],
                dumps(enriched_alert)
            ))
            
            enriched_alert_id = cur.fetchone()[0]

            # 2. Create and store the embedding for this new alert
            if embedding_model:
                fingerprint = f"Anomaly in {enriched_alert['application']}: {enriched_alert['metric_name']}."
                alert_embedding = embedding_model.encode(fingerprint)
                
                cur.execute("""
                    INSERT INTO historical_anomaly_embeddings (anomaly_id, embedding)
                    VALUES (%s, %s)
                """, (enriched_alert['source_anomaly_id'], alert_embedding.tobytes()))
            
            # 3. Mark the original anomaly as processed
            cur.execute("""
                UPDATE anomaly_detections 
                SET acknowledged = TRUE 
                WHERE id = %s
            """, (enriched_alert['source_anomaly_id'],))
            
            logger.info(f"Stored and indexed enriched alert for anomaly ID {enriched_alert['source_anomaly_id']}")
    except Exception as e:
        logger.error(f"Error storing enriched alert: {e}")

async def process_anomalies():
    """Main loop to fetch, process, and enrich anomalies."""
    logger.info("Starting Anomaly Detection Service")
    
    while True:
        try:
            unprocessed = fetch_unprocessed_anomalies()
            
            if not unprocessed:
                await asyncio.sleep(30) # Wait 30 seconds if no new anomalies
                continue
            
            logger.info(f"Found {len(unprocessed)} unprocessed anomalies. Starting enrichment.")
            
            for anomaly in unprocessed:
                logger.info(f"Processing anomaly ID {anomaly['id']} for {anomaly['application']}/{anomaly['metric_name']}")
                
                # 1. Correlate with other metrics
                correlated_metrics = get_correlated_metrics(anomaly)
                
                # 2. Get contextual data
                context = get_contextual_data(anomaly)
                
                # 3. Find historical matches
                historical_matches = find_historical_matches(anomaly, correlated_metrics)
                
                # 4. Calculate severity
                severity = calculate_severity(anomaly, correlated_metrics, historical_matches)
                
                # 5. Assemble the final enriched alert
                enriched_alert = {
                    "source_anomaly_id": anomaly['id'],
                    "application": anomaly['application'],
                    "metric_name": anomaly['metric_name'],
                    "detection_time": anomaly['detection_time'],
                    "severity": severity,
                    "original_anomaly": anomaly,
                    "correlated_metrics": correlated_metrics,
                    "context": context,
                    "historical_matches": historical_matches
                }
                
                # 6. Store the result and mark as processed
                store_and_index_enriched_alert(enriched_alert)
                
                # Small delay between processing each anomaly
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main processing loop: {e}")
            await asyncio.sleep(60) # Wait a minute before retrying after an error

async def main():
    """Main function to run the service"""
    logger.info("Starting Anomaly Detection Service")
    
    # Create database tables if they don't exist
    create_tables()
    
    # Start the main processing loop
    await process_anomalies()

if __name__ == "__main__":
    # To make this example runnable, create a dummy deployment log
    if not os.path.exists('simulated_deployment_log.json'):
        with open('simulated_deployment_log.json', 'w') as f:
            json.dump([{
                "app": "web_server",
                "version": "v1.4.1",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
            }], f)

    asyncio.run(main())
