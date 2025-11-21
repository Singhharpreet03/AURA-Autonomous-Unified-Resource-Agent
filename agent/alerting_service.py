import asyncio
import json
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("alerting_service")

# Database connection parameters (same as other services)
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = int(os.environ.get('DB_PORT'))
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_NAME = os.environ.get('DB_NAME')

# Alerting configuration
ALERT_CONFIG = {
    'enabled': os.getenv('ALERT_ENABLED', 'True').lower() == 'true',
    'channels': os.getenv('ALERT_CHANNELS', 'console,email,slack').split(','),
    'email': {
        'enabled': os.getenv('ALERT_ENABLED', 'True').lower() == 'true',
        'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', 587)),
        'username': os.getenv('EMAIL_USERNAME', 'your_email@gmail.com'),
        'password': os.getenv('EMAIL_PASSWORD', 'your_app_password'),
        'from': os.getenv('EMAIL_FROM', 'your_email@gmail.com'),
        'to': os.getenv('EMAIL_TO', 'admin@example.com,ops@example.com').split(',')
    },
    'slack': {
        'enabled': os.getenv('ALERT_ENABLED', 'True').lower() == 'true',
        'webhook_url': os.getenv('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'),
        'channel': os.getenv('SLACK_CHANNEL', '#alerts')
    },
    'rate_limit': {
        'minutes': int(os.getenv('ALERT_RATE_LIMIT_MINUTES', 5)),
        'max_alerts_per_hour': int(os.getenv('ALERT_MAX_ALERTS_PER_HOUR', 10))
    }
}

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

class AlertingService:
    def __init__(self):
        self.config = ALERT_CONFIG
        self.last_alerts = {}  # Track last alerts to avoid spamming
        self.alert_counts = {}  # Track alert counts for rate limiting
        
    async def check_for_new_alerts(self):
        """Check for new enriched alerts that haven't been notified yet"""
        try:
            with DatabaseConnection() as conn:
                cur = conn.cursor()
                
                # Get alerts that haven't been sent yet
                cur.execute("""
                    SELECT id, application, metric_name, detection_time, severity, enriched_details
                    FROM enriched_alerts
                    WHERE notified = FALSE
                    ORDER BY detection_time DESC
                    LIMIT 20
                """)
                
                columns = [desc[0] for desc in cur.description]
                alerts = [dict(zip(columns, row)) for row in cur.fetchall()]
                
                # Process each alert
                for alert in alerts:
                    await self.process_alert(alert)
                    
                    # Mark the alert as notified
                    cur.execute("""
                        UPDATE enriched_alerts
                        SET notified = TRUE, notified_at = NOW()
                        WHERE id = %s
                    """, (alert['id'],))
                
                conn.commit()
                logger.info(f"Processed {len(alerts)} new alerts")
                
        except Exception as e:
            logger.error(f"Error checking for new alerts: {e}")
    
    async def process_alert(self, alert: Dict):
        """Process a single alert and send notifications"""
        application = alert['application']
        severity = alert['severity']
        
        # Check rate limiting
        if not self.should_send_alert(application, severity):
            logger.info(f"Rate limiting alert for {application}")
            return
        
        # Update alert tracking
        self.update_alert_tracking(application)
        
        # Prepare alert message
        alert_data = {
            "id": alert['id'],
            "application": application,
            "metric_name": alert['metric_name'],
            "detection_time": alert['detection_time'].isoformat() if alert['detection_time'] else None,
            "severity": severity,
            "enriched_details": alert['enriched_details'] or {}
        }
        
        # Generate contextual message
        message = self.generate_alert_message(alert_data)
        
        # Send through all configured channels
        for channel in self.config['channels']:
            if channel == 'console':
                self.send_console_alert(message, alert_data)
            elif channel == 'email' and self.config['email']['enabled']:
                await self.send_email_alert(message, alert_data)
            elif channel == 'slack' and self.config['slack']['enabled']:
                await self.send_slack_alert(message, alert_data)
    
    def generate_alert_message(self, alert_data: Dict) -> str:
        """Generate a contextual alert message"""
        enriched_details = alert_data.get('enriched_details', {})
        correlated_metrics = enriched_details.get('correlated_metrics', [])
        context = enriched_details.get('context', {})
        historical_matches = enriched_details.get('historical_matches', [])
        
        # Base message
        message = f"""
        AURA Alert: Anomaly Detected in {alert_data['application']}
        
        Timestamp: {alert_data['detection_time']}
        Severity: {alert_data['severity'].upper()}
        Metric: {alert_data['metric_name']}
        
        """
        
        # Add correlated metrics if available
        if correlated_metrics:
            message += "Correlated Metrics:\n"
            for metric in correlated_metrics:
                message += f"- {metric.get('metric_name', 'Unknown')}: {metric.get('value', 'N/A')}\n"
            message += "\n"
        
        # Add context if available
        if context:
            message += "Context:\n"
            if 'recent_deployment' in context:
                deploy = context['recent_deployment']
                message += f"- Recent deployment: Version {deploy.get('version', 'Unknown')} at {deploy.get('timestamp', 'Unknown')}\n"
            message += "\n"
        
        # Add historical matches if available
        if historical_matches:
            message += f"Historical Patterns: Found {len(historical_matches)} similar past anomalies\n"
            if historical_matches:
                top_match = historical_matches[0]
                message += f"- Top match similarity: {top_match.get('similarity', 0):.2f}\n"
            message += "\n"
        
        message += "Please investigate this anomaly."
        
        return message
    
    def should_send_alert(self, application: str, severity: str) -> bool:
        """Check if we should send an alert based on rate limiting"""
        now = datetime.now()
        
        # Check if we've recently sent an alert for this application
        if application in self.last_alerts:
            time_diff = (now - self.last_alerts[application]).total_seconds()
            if time_diff < self.config['rate_limit']['minutes'] * 60:
                # Allow critical alerts to bypass rate limiting
                if severity != 'critical':
                    return False
        
        # Check hourly limit
        hour_key = f"{application}_{now.hour}"
        if hour_key in self.alert_counts:
            if self.alert_counts[hour_key] >= self.config['rate_limit']['max_alerts_per_hour']:
                # Allow critical alerts to bypass rate limiting
                if severity != 'critical':
                    return False
        
        return True
    
    def update_alert_tracking(self, application: str):
        """Update alert tracking for rate limiting"""
        now = datetime.now()
        self.last_alerts[application] = now
        
        # Update hourly count
        hour_key = f"{application}_{now.hour}"
        if hour_key not in self.alert_counts:
            self.alert_counts[hour_key] = 0
        self.alert_counts[hour_key] += 1
        
        # Clean old entries (older than 2 hours)
        current_hour = now.hour
        for key in list(self.alert_counts.keys()):
            app, hour = key.split('_', 1)
            hour = int(hour)
            if (current_hour - hour) % 24 > 2:
                del self.alert_counts[key]
    
    def send_console_alert(self, message: str, alert_data: Dict):
        """Send alert to console"""
        if alert_data['severity'] == 'critical':
            logger.error(f"CRITICAL ALERT: {message}")
        elif alert_data['severity'] == 'high':
            logger.warning(f"HIGH ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")
    
    async def send_email_alert(self, message: str, alert_data: Dict):
        """Send email alert"""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            
            # Set subject based on severity
            severity_prefix = {
                'critical': '[CRITICAL]',
                'high': '[HIGH]',
                'medium': '[MEDIUM]',
                'low': '[LOW]'
            }.get(alert_data['severity'], '[ALERT]')
            
            msg['Subject'] = f"{severity_prefix} AURA Alert: Anomaly in {alert_data['application']}"
            
            # Create HTML version of the message
            html_message = f"""
            <html>
            <body>
                <h2>AURA Alert: Anomaly Detected in {alert_data['application']}</h2>
                <p><strong>Timestamp:</strong> {alert_data['detection_time']}</p>
                <p><strong>Severity:</strong> <span style="color: {'red' if alert_data['severity'] == 'critical' else 'orange' if alert_data['severity'] == 'high' else 'blue'}">{alert_data['severity'].upper()}</span></p>
                <p><strong>Metric:</strong> {alert_data['metric_name']}</p>
                
                <h3>Details:</h3>
                <pre>{message}</pre>
                
                <p>This is an automated alert from the AURA system.</p>
            </body>
            </html>
            """
            
            # Attach both plain text and HTML versions
            msg.attach(MIMEText(message, 'plain'))
            msg.attach(MIMEText(html_message, 'html'))
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
                
            logger.info(f"Email alert sent for {alert_data['application']}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def send_slack_alert(self, message: str, alert_data: Dict):
        """Send Slack alert"""
        try:
            slack_config = self.config['slack']
            
            # Set color based on severity
            color = {
                'critical': 'danger',
                'high': 'warning',
                'medium': 'good',
                'low': '#439FE0'
            }.get(alert_data['severity'], 'warning')
            
            # Create rich Slack message
            payload = {
                "channel": slack_config['channel'],
                "username": "AURA Alert Bot",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"AURA Alert: Anomaly in {alert_data['application']}",
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert_data['severity'].upper(),
                                "short": True
                            },
                            {
                                "title": "Metric",
                                "value": alert_data['metric_name'],
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": alert_data['detection_time'],
                                "short": True
                            }
                        ],
                        "text": message,
                        "footer": "AURA System",
                        "ts": datetime.now().timestamp()
                    }
                ]
            }
            
            response = requests.post(
                slack_config['webhook_url'],
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for {alert_data['application']}")
            else:
                logger.error(f"Failed to send Slack alert: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def run(self):
        """Run the alerting service"""
        if not self.config['enabled']:
            logger.info("Alerting service is disabled")
            return
            
        logger.info("Starting alerting service")
        
        # Create the notified column if it doesn't exist
        try:
            with DatabaseConnection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    ALTER TABLE enriched_alerts 
                    ADD COLUMN IF NOT EXISTS notified BOOLEAN DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS notified_at TIMESTAMP
                """)
                conn.commit()
                logger.info("Database schema updated for alerting")
        except Exception as e:
            logger.error(f"Error updating database schema: {e}")
        
        # Main loop
        while True:
            await self.check_for_new_alerts()
            await asyncio.sleep(30)  # Check for new alerts every 30 seconds

if __name__ == "__main__":
    service = AlertingService()
    asyncio.run(service.run())
