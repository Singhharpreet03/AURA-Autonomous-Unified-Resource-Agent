import asyncio
import json
import psycopg2
import time
import psutil
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging
import pickle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import platform
import subprocess
from json_utils import dumps 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection parameters
DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT'))
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

# Interval for collecting metrics (in seconds)
COLLECTION_INTERVAL = 15
METRICS_FILE = 'metrics.json'
MODEL_FILE = 'collection_model.pkl'
MODEL_TRAINING_INTERVAL = 60  # Train the model every 60 seconds
MIN_COLLECTION_INTERVAL = 5   # Minimum collection interval in seconds
MAX_COLLECTION_INTERVAL = 30  # Maximum collection interval in seconds

# Global variables to track last training time and disk/network state for IOPS
last_training_time = 0
last_disk_counters = None
last_network_counters = None
last_collection_time = 0

THRESHOLDS = {
    'cpu_usage': float(os.getenv('CPU_USAGE_THRESHOLD', 70)),
    'memory_usage': float(os.getenv('MEMORY_USAGE_THRESHOLD', 90)),
    'gpu_usage': float(os.getenv('GPU_USAGE_THRESHOLD', 80)),
    'disk_read_iops': float(os.getenv('DISK_READ_IOPS_THRESHOLD', 5000)),
    'disk_write_iops': float(os.getenv('DISK_WRITE_IOPS_THRESHOLD', 5000)),
}

# Check for NVIDIA GPU availability at startup
has_nvidia = False
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    has_nvidia = device_count > 0
    pynvml.nvmlShutdown()
    logging.info(f"NVIDIA GPU {'detected' if has_nvidia else 'not detected'}. GPU metrics collection {'enabled' if has_nvidia else 'disabled'}.")
except (ImportError, OSError, Exception) as e:
    logging.info(f"No NVIDIA GPU available: {e}. GPU metrics collection disabled.")

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

def create_metrics_table():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id SERIAL PRIMARY KEY,
                time TIMESTAMP,
                application VARCHAR(100),
                cpu_usage FLOAT,
                cpu_cores INTEGER,
                cpu_freq FLOAT,
                memory_usage FLOAT,
                memory_total_mb FLOAT,
                memory_used_mb FLOAT,
                memory_available_mb FLOAT,
                gpu_usage FLOAT,
                gpu_memory_usage FLOAT,
                gpu_vram_used_mb FLOAT,
                gpu_vram_total_mb FLOAT,
                disk_read_iops FLOAT,
                disk_write_iops FLOAT,
                disk_read_bytes FLOAT,
                disk_write_bytes FLOAT,
                network_packets_sent_per_sec FLOAT,
                network_packets_recv_per_sec FLOAT,
                network_bytes_sent_per_sec FLOAT,
                network_bytes_recv_per_sec FLOAT,
                network_dropin BIGINT,
                network_dropout BIGINT,
                platform_specific JSONB,
                custom_metrics JSONB
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logging.info("Metrics table created or verified successfully.")
    except Exception as e:
        logging.error(f"Error creating metrics table: {e}")
        raise

def insert_metrics(metrics):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO metrics (
                time, application, cpu_usage, cpu_cores, cpu_freq,
                memory_usage, memory_total_mb, memory_used_mb, memory_available_mb,
                gpu_usage, gpu_memory_usage, gpu_vram_used_mb, gpu_vram_total_mb,
                disk_read_iops, disk_write_iops, disk_read_bytes, disk_write_bytes,
                network_packets_sent_per_sec, network_packets_recv_per_sec,
                network_bytes_sent_per_sec, network_bytes_recv_per_sec,
                network_dropin, network_dropout, platform_specific, custom_metrics
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            metrics['time'],
            metrics['application'],
            metrics['cpu']['usage'],
            metrics['cpu']['cores'],
            metrics['cpu']['freq'],
            metrics['memory']['percent'],
            metrics['memory']['total_mb'],
            metrics['memory']['used_mb'],
            metrics['memory']['available_mb'],
            metrics['gpu'][0]['gpu_utilization'] if metrics['gpu'] else None,
            metrics['gpu'][0]['memory_utilization'] if metrics['gpu'] else None,
            metrics['gpu'][0]['vram_used_mb'] if metrics['gpu'] else None,
            metrics['gpu'][0]['vram_total_mb'] if metrics['gpu'] else None,
            metrics['disk']['read_iops'],
            metrics['disk']['write_iops'],
            metrics['disk']['read_bytes'],
            metrics['disk']['write_bytes'],
            metrics['network']['packets_sent_per_sec'],
            metrics['network']['packets_recv_per_sec'],
            metrics['network']['bytes_sent_per_sec'],
            metrics['network']['bytes_recv_per_sec'],
            metrics['network']['dropin'],
            metrics['network']['dropout'],
            dumps(metrics['platform_specific']),
            dumps(metrics['custom'])
        ))
        conn.commit()
        cur.close()
        conn.close()
        logging.info("Metrics inserted successfully.")
    except Exception as e:
        logging.error(f"Error inserting metrics: {e}")


def insert_threshold_anomaly(application: str, metric: str, value: float, ts):
    """Create a critical anomaly row straight from threshold breach."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO anomaly_detections
                (application, metric_name, detection_time, value,
                 anomaly_score, severity, details, acknowledged)
            VALUES (%s, %s, %s, %s,  1.0, 'critical', %s, false)
            ON CONFLICT DO NOTHING
        """, (
            application,
            metric,
            ts,
            float(value),
            dumps({'source': 'threshold', 'threshold': THRESHOLDS[metric]})
        ))
        conn.commit(); cur.close(); conn.close()
        logging.warning("THRESHOLD BREACH %s/%s = %s %%", application, metric, value)
    except Exception as e:
        logging.error("Fast-path insert failed: %s", e)

def train_model(current_time):
    global last_training_time
    if current_time - last_training_time < MODEL_TRAINING_INTERVAL:
        return  # Skip if not enough time has passed

    try:
        # Load existing data robustly
        data = []
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid line due to JSON error: {e}")
                        continue

        # Extract CPU usage data
        if not data:
            logging.warning("No valid data to train the model. Skipping training.")
            return

        # ---- CORRECTED LINE ----
        # Add isinstance(entry, dict) to ensure we only process dictionary objects
        cpu_usage = [entry['cpu']['usage'] for entry in data if isinstance(entry, dict) and 'cpu' in entry and 'usage' in entry['cpu']]
        
        if len(cpu_usage) < 24:  # Assuming 24 data points for two full cycles
            logging.warning(f"Not enough data to train the model ({len(cpu_usage)} points). Skipping training.")
            return

        cpu_series = pd.Series(cpu_usage)

        # Train the model
        model = ExponentialSmoothing(cpu_series, trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()

        # Save the model using pickle
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_fit, f)
        last_training_time = current_time
        logging.info("Model trained and saved successfully.")
    except Exception as e:
        logging.error(f"Error training model: {e}")

def predict_collection_interval():
    try:
        if not os.path.exists(MODEL_FILE):
            logging.info("No model file found. Using default interval.")
            return COLLECTION_INTERVAL

        with open(MODEL_FILE, 'rb') as f:
            model_fit = pickle.load(f)

        # ---- guard against empty model ----
        if model_fit.model.endog.size == 0:
            logging.warning("Model was trained on empty data. Using default interval.")
            return COLLECTION_INTERVAL

        # ---- CORRECTED FORECAST LOGIC ----
        forecast_result = model_fit.forecast(steps=1)
        
        # Check if the forecast returned any values
        if forecast_result.empty:
            logging.warning("Model forecast is empty. Using default interval.")
            return COLLECTION_INTERVAL
            
        # Access the first value by its integer position using .iloc[0]
        prediction = forecast_result.iloc[0]

        if prediction > 80:  # High CPU usage
            adjusted = max(MIN_COLLECTION_INTERVAL, COLLECTION_INTERVAL / 2)
        elif prediction < 20:  # Low CPU usage
            adjusted = min(MAX_COLLECTION_INTERVAL, COLLECTION_INTERVAL * 2)
        else:
            adjusted = COLLECTION_INTERVAL

        logging.info(f"Predicted CPU: {prediction:.2f}%, adjusted interval: {adjusted}s")
        return adjusted
    except Exception:
        logging.exception("Error predicting collection interval")   # full traceback
        return COLLECTION_INTERVAL

def collect_custom_metrics():
    # Placeholder for future integration with other apps/custom metrics
    return {}

def get_platform_specific_metrics():
    """Get metrics that are specific to the platform"""
    platform_specific = {}
    
    try:
        if platform.system() == 'Windows':
            # Windows-specific metrics
            try:
                import wmi
                c = wmi.WMI()
                
                # Get Windows-specific CPU metrics
                cpu_info = c.Win32_Processor()[0]
                platform_specific['windows_cpu'] = {
                    'name': cpu_info.Name,
                    'manufacturer': cpu_info.Manufacturer,
                    'max_clock_speed': cpu_info.MaxClockSpeed,
                    'current_clock_speed': cpu_info.CurrentClockSpeed,
                    'load_percentage': cpu_info.LoadPercentage
                }
                
                # Get Windows-specific memory metrics
                os_info = c.Win32_OperatingSystem()[0]
                platform_specific['windows_memory'] = {
                    'total_visible_memory_size': os_info.TotalVisibleMemorySize,
                    'free_physical_memory': os_info.FreePhysicalMemory,
                    'total_virtual_memory_size': os_info.TotalVirtualMemorySize,
                    'free_virtual_memory': os_info.FreeVirtualMemory
                }
                
                # Get Windows-specific disk metrics
                disk_drives = c.Win32_LogicalDisk(DriveType=3)  # Local disks
                platform_specific['windows_disk'] = []
                for disk in disk_drives:
                    platform_specific['windows_disk'].append({
                        'device_id': disk.DeviceID,
                        'size': disk.Size,
                        'free_space': disk.FreeSpace,
                        'volume_name': disk.VolumeName
                    })
                
                # Get Windows-specific network metrics
                network_configs = c.Win32_NetworkAdapterConfiguration(IPEnabled=True)
                platform_specific['windows_network'] = []
                for config in network_configs:
                    platform_specific['windows_network'].append({
                        'description': config.Description,
                        'ip_address': config.IPAddress,
                        'default_ip_gateway': config.DefaultIPGateway,
                        'dns_server': config.DNSServerSearchOrder
                    })
                
                # Get Windows-specific process metrics
                processes = c.Win32_Process()
                platform_specific['windows_processes'] = {
                    'count': len(processes),
                    'top_cpu': []
                }
                
                valid_processes = [proc for proc in processes if proc.PercentProcessorTime is not None]
                for proc in sorted(valid_processes, key=lambda p: p.PercentProcessorTime, reverse=True)[:5]:
                    platform_specific['windows_processes']['top_cpu'].append({
                        'name': proc.Name,
                        'process_id': proc.ProcessId,
                        'cpu_usage': proc.PercentProcessorTime,
                        'memory_usage': proc.WorkingSetSize / (1024 * 1024)  # Convert to MB
                    })
                
            except ImportError:
                logging.warning("WMI module not available. Windows-specific metrics disabled.")
            except Exception as e:
                logging.warning(f"Error collecting Windows-specific metrics: {e}")
                
        elif platform.system() == 'Linux':
            # Linux-specific metrics
            try:
                # Get CPU temperature if available
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temp = float(f.read().strip()) / 1000.0  # Convert from millidegrees to degrees
                        platform_specific['linux_cpu_temp'] = temp
                except:
                    pass
                
                # Get detailed CPU info
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read()
                        platform_specific['linux_cpuinfo'] = cpuinfo
                except:
                    pass
                
                # Get memory info
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = {}
                        for line in f:
                            key, value = line.split(':', 1)
                            meminfo[key.strip()] = value.strip()
                        platform_specific['linux_meminfo'] = meminfo
                except:
                    pass
                
                # Get disk info
                try:
                    result = subprocess.run(['df', '-h'], capture_output=True, text=True)
                    if result.returncode == 0:
                        platform_specific['linux_disk'] = result.stdout
                except:
                    pass
                
                # Get network interface info
                try:
                    result = subprocess.run(['ip', 'addr'], capture_output=True, text=True)
                    if result.returncode == 0:
                        platform_specific['linux_network'] = result.stdout
                except:
                    pass
                
                # Get top processes
                try:
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')[1:]  # Skip header
                        processes = []
                        for line in lines[:10]:  # Get top 10
                            parts = line.split(None, 10)
                            if len(parts) >= 11:
                                processes.append({
                                    'user': parts[0],
                                    'pid': parts[1],
                                    'cpu': parts[2],
                                    'mem': parts[3],
                                    'command': parts[10]
                                })
                        platform_specific['linux_processes'] = processes
                except:
                    pass
                
            except Exception as e:
                logging.warning(f"Error collecting Linux-specific metrics: {e}")
    except Exception as e:
        logging.error(f"Error collecting platform-specific metrics: {e}")
    
    return platform_specific

async def collect_metrics():
    global last_training_time, last_disk_counters, last_network_counters, last_collection_time
    # Create metrics table if it doesn't exist
    create_metrics_table()

    while True:
        current_time = time.time()

        # Predict the next collection interval
        interval = predict_collection_interval()

        # Collect CPU metrics
        cpu = psutil.cpu_percent(interval=1)
        cpu_cores = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        cpu_metrics = {
            'usage': cpu,
            'cores': cpu_cores,
            'freq': cpu_freq
        }

        # Collect memory (RAM) metrics
        memory = psutil.virtual_memory()
        memory_metrics = {
            'used_mb': memory.used / (1024 * 1024),  # Convert to MB
            'available_mb': memory.available / (1024 * 1024),
            'percent': memory.percent,
            'total_mb': memory.total / (1024 * 1024)
        }

        # Collect disk IOPS (read/write operations per second)
        disk_counters = psutil.disk_io_counters()
        disk_metrics = {
            'read_count': 0, 
            'write_count': 0, 
            'read_iops': 0, 
            'write_iops': 0,
            'read_bytes': 0,
            'write_bytes': 0
        }
        if disk_counters:
            disk_metrics.update(disk_counters._asdict())
            if last_disk_counters and last_collection_time:
                time_delta = current_time - last_collection_time
                if time_delta > 0:
                    disk_metrics['read_iops'] = (disk_counters.read_count - last_disk_counters.read_count) / time_delta
                    disk_metrics['write_iops'] = (disk_counters.write_count - last_disk_counters.write_count) / time_delta
                    disk_metrics['read_bytes'] = disk_counters.read_bytes - last_disk_counters.read_bytes
                    disk_metrics['write_bytes'] = disk_counters.write_bytes - last_disk_counters.write_bytes
            last_disk_counters = disk_counters

        # Collect network IOPS and packet drops
        network_counters = psutil.net_io_counters(pernic=False)  # Sum across all interfaces
        network_metrics = {
            'packets_sent': 0, 
            'packets_recv': 0, 
            'bytes_sent': 0, 
            'bytes_recv': 0,
            'packets_sent_per_sec': 0, 
            'packets_recv_per_sec': 0,
            'bytes_sent_per_sec': 0,
            'bytes_recv_per_sec': 0,
            'dropin': 0, 
            'dropout': 0
        }
        if network_counters:
            network_metrics.update(network_counters._asdict())
            if last_network_counters and last_collection_time:
                time_delta = current_time - last_collection_time
                if time_delta > 0:
                    network_metrics['packets_sent_per_sec'] = (network_counters.packets_sent - last_network_counters.packets_sent) / time_delta
                    network_metrics['packets_recv_per_sec'] = (network_counters.packets_recv - last_network_counters.packets_recv) / time_delta
                    network_metrics['bytes_sent_per_sec'] = (network_counters.bytes_sent - last_network_counters.bytes_sent) / time_delta
                    network_metrics['bytes_recv_per_sec'] = (network_counters.bytes_recv - last_network_counters.bytes_recv) / time_delta
            last_network_counters = network_counters

        # Collect detailed network interface metrics for packet drops
        try:
            net_if_addrs = psutil.net_if_addrs()
            net_if_stats = psutil.net_if_stats()
            
            interface_metrics = {}
            for interface_name, addresses in net_if_addrs.items():
                if interface_name in net_if_stats:
                    interface_metrics[interface_name] = {
                        'isup': net_if_stats[interface_name].isup,
                        'duplex': net_if_stats[interface_name].duplex,
                        'speed': net_if_stats[interface_name].speed,
                        'mtu': net_if_stats[interface_name].mtu
                    }
                    
                    # Get IP addresses
                    ips = []
                    for addr in addresses:
                        if addr.family == 2:  # AF_INET
                            ips.append(addr.address)
                    interface_metrics[interface_name]['ips'] = ips
            
            # Get per-interface network I/O counters
            net_io_counters = psutil.net_io_counters(pernic=True)
            for interface_name, counters in net_io_counters.items():
                if interface_name in interface_metrics:
                    interface_metrics[interface_name]['io_counters'] = counters._asdict()
            
            network_metrics['interfaces'] = interface_metrics
        except Exception as e:
            logging.warning(f"Error collecting detailed network metrics: {e}")

        # Collect GPU VRAM if NVIDIA GPU is available
        gpu_metrics = []
        if has_nvidia:
            init_success = False
            try:
                import pynvml
                pynvml.nvmlInit()
                init_success = True
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = None
                    
                    # Get power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    except:
                        power = None
                    
                    # Get clock speeds
                    try:
                        graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    except:
                        graphics_clock = None
                        memory_clock = None
                    
                    # Get GPU name
                    try:
                        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    except:
                        name = "Unknown"
                    
                    gpu_metrics.append({
                        'id': i,
                        'name': name,
                        'gpu_utilization': util_rates.gpu,
                        'memory_utilization': util_rates.memory,
                        'vram_used_mb': mem_info.used / (1024 * 1024),  # Convert to MB
                        'vram_total_mb': mem_info.total / (1024 * 1024),
                        'vram_free_mb': mem_info.free / (1024 * 1024),
                        'temperature': temp,
                        'power_usage_watts': power,
                        'graphics_clock_mhz': graphics_clock,
                        'memory_clock_mhz': memory_clock
                    })
                if gpu_metrics:
                    logging.info(f"Collected metrics for {len(gpu_metrics)} GPU(s).")
            except (ImportError, OSError, Exception) as e:
                logging.warning(f"GPU metrics unavailable: {e}")
            finally:
                if init_success:
                    try:
                        pynvml.nvmlShutdown()
                    except:
                        pass

        # Collect custom metrics
        custom_metrics = collect_custom_metrics()

        # Collect platform-specific metrics
        platform_specific = get_platform_specific_metrics()

        # Combine all metrics
        metrics = {
            'time': pd.Timestamp.now(),
            'application': 'monitor',  # Replace with your application name
            'cpu': cpu_metrics,
            'memory': memory_metrics,
            'disk': disk_metrics,
            'network': network_metrics,
            'gpu': gpu_metrics if gpu_metrics else None,
            'platform_specific': platform_specific,
            'custom': custom_metrics
        }

        # Insert metrics into PostgreSQL
        insert_metrics(metrics)

        # Save metrics to file for model training
        try:
            metrics['time'] = metrics['time'].isoformat()
            with open(METRICS_FILE, 'a') as f:
                f.write(dumps(metrics) + '\n')
        except Exception as e:
            logging.error(f"Error writing to metrics file: {e}")

        # Train the model periodically
        train_model(current_time)

        # Update last collection time
        last_collection_time = current_time

        try:
            app = metrics['application']          # 'monitor'
            ts  = metrics['time']

            # CPU
            if metrics['cpu']['usage'] >= THRESHOLDS['cpu_usage']:
                insert_threshold_anomaly(app, 'cpu_usage', metrics['cpu']['usage'], ts)

            # Memory
            if metrics['memory']['percent'] >= THRESHOLDS['memory_usage']:
                insert_threshold_anomaly(app, 'memory_usage', metrics['memory']['percent'], ts)

            # GPU (if present)
            if metrics['gpu']:
                gpu_util = metrics['gpu'][0]['gpu_utilization']
                if gpu_util is not None and gpu_util >= THRESHOLDS['gpu_usage']:
                    insert_threshold_anomaly(app, 'gpu_usage', gpu_util, ts)

            # Disk IOPS
            if metrics['disk']['read_iops'] >= THRESHOLDS['disk_read_iops']:
                insert_threshold_anomaly(app, 'disk_read_iops', metrics['disk']['read_iops'], ts)
            if metrics['disk']['write_iops'] >= THRESHOLDS['disk_write_iops']:
                insert_threshold_anomaly(app, 'disk_write_iops', metrics['disk']['write_iops'], ts)

        except Exception as e:
            logging.error("Fast-path threshold check failed: %s", e)


        # Wait for the next collection interval
        logging.info(f"Next collection interval: {interval} seconds")
        await asyncio.sleep(interval)

if __name__ == "__main__":
    # Initialize last_training_time and last_collection_time
    last_training_time = time.time()
    last_collection_time = time.time()
    asyncio.run(collect_metrics())