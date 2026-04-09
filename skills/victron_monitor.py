"""
HelioAi Monitor — Solar + System Data Collector

Monitors Victron MPPT battery via BLE every 10 minutes,
GPU/system metrics every 2 minutes, stores in SQLite.
"""

import asyncio
import sqlite3
import subprocess
import threading
import time
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
from contextlib import contextmanager
import sys
sys.path.insert(0, '.')

from victron_ble.scanner import Scanner
from victron_ble.devices import SolarCharger


# Database setup
DB_PATH = os.environ.get('DB_PATH', "C:/Users/aaron/victron_monitor.db")
DEVICE_ADDRESS = os.environ.get('DEVICE_ADDRESS', "CE:CE:F0:AD:16:02")
ADVERTISEMENT_KEY = os.environ.get('ADVERTISEMENT_KEY', "71c0587577ba16767d312f85aa1b7f7e")


@contextmanager
def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Initialize database with readings and system_readings tables."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                battery_voltage REAL,
                battery_current REAL,
                solar_power INTEGER,
                charge_state TEXT,
                charger_error TEXT,
                model_name TEXT,
                yield_today REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON readings(timestamp DESC)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                gpu_temp REAL,
                gpu_util REAL,
                vram_used INTEGER,
                vram_total INTEGER,
                gpu_power REAL,
                gpu_power_limit REAL,
                battery_percent INTEGER,
                battery_status TEXT,
                uptime_hours REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_timestamp
            ON system_readings(timestamp DESC)
        """)


def save_reading(data: Dict[str, Any]) -> None:
    """Save a Victron reading to the database."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO readings (
                battery_voltage, battery_current, solar_power,
                charge_state, charger_error, model_name, yield_today
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('battery_voltage'),
            data.get('battery_current'),
            data.get('solar_power'),
            str(data.get('charge_state', '')),
            str(data.get('charger_error', '')),
            data.get('model_name'),
            data.get('yield_today')
        ))


def save_system_reading(data: Dict[str, Any]) -> None:
    """Save a system reading to the database."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO system_readings (
                gpu_temp, gpu_util, vram_used, vram_total,
                gpu_power, gpu_power_limit, battery_percent,
                battery_status, uptime_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('gpu_temp'),
            data.get('gpu_util'),
            data.get('vram_used'),
            data.get('vram_total'),
            data.get('gpu_power'),
            data.get('gpu_power_limit'),
            data.get('battery_percent'),
            data.get('battery_status'),
            data.get('uptime_hours'),
        ))


def get_readings(hours: int = 24) -> List[Dict[str, Any]]:
    """Get Victron readings from the last N hours."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM readings
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp ASC
        """, (hours,)).fetchall()
        return [dict(row) for row in rows]


def get_latest_reading() -> Dict[str, Any]:
    """Get the most recent Victron reading."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM readings
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchone()
        return dict(row) if row else None


def get_stats() -> Dict[str, Any]:
    """Get statistics from the Victron readings."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as count,
                MIN(battery_voltage) as min_voltage,
                MAX(battery_voltage) as max_voltage,
                AVG(battery_voltage) as avg_voltage
            FROM readings
            WHERE battery_voltage IS NOT NULL
        """).fetchone()

        stats = {
            'total_readings': row['count'],
            'min_voltage': row['min_voltage'],
            'max_voltage': row['max_voltage'],
            'avg_voltage': row['avg_voltage']
        }

        yield_row = conn.execute("""
            SELECT yield_today
            FROM readings
            WHERE yield_today IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchone()

        if yield_row:
            stats['yield_today'] = yield_row['yield_today']

        return stats


def collect_system_data() -> Dict[str, Any]:
    """Collect GPU, laptop battery, and uptime data."""
    data = {}

    # GPU data via nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) == 6:
                data['gpu_temp'] = float(parts[0]) if parts[0] != '[N/A]' else None
                data['gpu_util'] = float(parts[1]) if parts[1] != '[N/A]' else None
                data['vram_used'] = int(parts[2]) if parts[2] != '[N/A]' else None
                data['vram_total'] = int(parts[3]) if parts[3] != '[N/A]' else None
                data['gpu_power'] = float(parts[4]) if parts[4] != '[N/A]' else None
                data['gpu_power_limit'] = float(parts[5]) if parts[5] != '[N/A]' else None
    except Exception as e:
        print(f"  GPU collect error: {e}")

    # Laptop battery via PowerShell
    try:
        ps = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             'Get-CimInstance Win32_Battery | Select-Object -Property EstimatedChargeRemaining,BatteryStatus | ConvertTo-Json'],
            capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW
        )
        if ps.returncode == 0 and ps.stdout.strip():
            import json
            batt = json.loads(ps.stdout.strip())
            data['battery_percent'] = batt.get('EstimatedChargeRemaining')
            status_map = {1: 'Discharging', 2: 'AC Power', 3: 'Fully Charged', 4: 'Low', 5: 'Critical'}
            data['battery_status'] = status_map.get(batt.get('BatteryStatus'), 'Unknown')
    except Exception as e:
        print(f"  Battery collect error: {e}")

    # Uptime via PowerShell
    try:
        ps = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '((Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime).TotalHours'],
            capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW
        )
        if ps.returncode == 0 and ps.stdout.strip():
            data['uptime_hours'] = round(float(ps.stdout.strip()), 1)
    except Exception as e:
        print(f"  Uptime collect error: {e}")

    return data


def get_system_latest() -> Dict[str, Any]:
    """Get the most recent system reading."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM system_readings
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchone()
        return dict(row) if row else None


def get_system_readings(hours: int = 24) -> List[Dict[str, Any]]:
    """Get system readings from the last N hours."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM system_readings
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp ASC
        """, (hours,)).fetchall()
        return [dict(row) for row in rows]


async def read_mppt_data() -> Dict[str, Any]:
    """Read data from MPPT via BLE."""
    charger = SolarCharger(advertisement_key=ADVERTISEMENT_KEY)
    scanner = Scanner(device_keys={DEVICE_ADDRESS.lower(): ADVERTISEMENT_KEY})

    data_result = {'error': None}

    def callback(device, data, advertisement):
        device_addr = device.address if hasattr(device, 'address') else str(device)
        if device_addr.lower() != DEVICE_ADDRESS.lower():
            return

        if isinstance(data, bytes):
            try:
                parsed_data = charger.parse(data)
                data_result.update({
                    'battery_voltage': parsed_data.get_battery_voltage(),
                    'battery_current': parsed_data.get_battery_charging_current(),
                    'solar_power': parsed_data.get_solar_power(),
                    'charge_state': parsed_data.get_charge_state(),
                    'charger_error': parsed_data.get_charger_error(),
                    'model_name': parsed_data.get_model_name(),
                    'yield_today': parsed_data.get_yield_today(),
                    'load_output': parsed_data.get_external_device_load(),
                })
            except Exception as e:
                data_result['error'] = str(e)

    scanner.callback = callback
    await scanner.start()
    await asyncio.sleep(3)
    await scanner.stop()

    if data_result.get('error'):
        raise Exception(data_result['error'])

    return data_result


class MonitorScheduler:
    """Background scheduler to collect readings."""

    def __init__(self, interval_minutes: int = 10, system_interval_minutes: int = 2):
        self.interval_minutes = interval_minutes
        self.system_interval_minutes = system_interval_minutes
        self.running = False
        self.victron_thread = None
        self.system_thread = None

    def collect_victron(self):
        """Collect and save a Victron BLE reading."""
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Collecting Victron reading...")
            data = asyncio.run(read_mppt_data())
            save_reading(data)
            voltage = data.get('battery_voltage')
            power = data.get('solar_power')
            print(f"  Battery: {voltage:.2f} V | Solar: {power} W")
        except Exception as e:
            print(f"  Victron error: {e}")

    def collect_system(self):
        """Collect and save system data."""
        try:
            data = collect_system_data()
            save_system_reading(data)
            gpu_t = data.get('gpu_temp', '?')
            gpu_p = data.get('gpu_power', '?')
            batt = data.get('battery_percent', '?')
            print(f"  GPU: {gpu_t}C {gpu_p}W | Battery: {batt}%")
        except Exception as e:
            print(f"  System collect error: {e}")

    def run_victron(self):
        """Victron scheduler loop."""
        self.running = True
        while self.running:
            self.collect_victron()
            # Sleep in 1-second intervals so we can stop promptly
            for _ in range(self.interval_minutes * 60):
                if not self.running:
                    return
                time.sleep(1)

    def run_system(self):
        """System data scheduler loop."""
        while self.running:
            self.collect_system()
            for _ in range(self.system_interval_minutes * 60):
                if not self.running:
                    return
                time.sleep(1)

    def start(self):
        """Start both schedulers in background threads."""
        self.running = True
        if self.victron_thread is None or not self.victron_thread.is_alive():
            self.victron_thread = threading.Thread(target=self.run_victron, daemon=True)
            self.victron_thread.start()
            print(f"Victron scheduler started (every {self.interval_minutes} min)")

        if self.system_thread is None or not self.system_thread.is_alive():
            self.system_thread = threading.Thread(target=self.run_system, daemon=True)
            self.system_thread.start()
            print(f"System scheduler started (every {self.system_interval_minutes} min)")

    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.victron_thread:
            self.victron_thread.join(timeout=5)
        if self.system_thread:
            self.system_thread.join(timeout=5)


# Global scheduler instance
scheduler = MonitorScheduler(interval_minutes=10, system_interval_minutes=2)


if __name__ == "__main__":
    init_db()
    scheduler.start()
    print("\nHelioAi Monitor running. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        scheduler.stop()
        print("Monitor stopped.")
