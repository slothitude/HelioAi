"""
HelioAi Dashboard — Flask web application

Displays solar, GPU, and system data with dark-themed UI.
"""

import os
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import sys
sys.path.insert(0, '.')

from skills.victron_monitor import (
    init_db, get_readings, get_latest_reading, get_stats,
    get_system_latest, get_system_readings, collect_system_data,
    save_system_reading,
    scheduler, read_mppt_data, save_reading
)


app = Flask(__name__)


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/readings')
def api_readings():
    """Get Victron readings as JSON for charts."""
    hours = int(request.args.get('hours', 24))
    readings = get_readings(hours=hours)
    data = {
        'labels': [r['timestamp'].replace('T', ' ') for r in readings],
        'voltage': [r['battery_voltage'] for r in readings],
        'current': [r['battery_current'] for r in readings],
        'power': [r['solar_power'] for r in readings],
    }
    return jsonify(data)


@app.route('/api/latest')
def api_latest():
    """Get latest Victron reading."""
    latest = get_latest_reading()
    if latest:
        latest['timestamp'] = latest['timestamp']
    return jsonify(latest)


@app.route('/api/stats')
def api_stats():
    """Get Victron statistics."""
    return jsonify(get_stats())


@app.route('/api/system')
def api_system():
    """Get latest system reading (GPU, battery, uptime)."""
    latest = get_system_latest()
    return jsonify(latest)


@app.route('/api/system/history')
def api_system_history():
    """Get system readings history for charts."""
    hours = int(request.args.get('hours', 24))
    readings = get_system_readings(hours=hours)
    data = {
        'labels': [r['timestamp'].replace('T', ' ') for r in readings],
        'gpu_temp': [r['gpu_temp'] for r in readings],
        'gpu_util': [r['gpu_util'] for r in readings],
        'vram_used': [r['vram_used'] for r in readings],
        'vram_total': [r['vram_total'] for r in readings],
        'gpu_power': [r['gpu_power'] for r in readings],
        'battery_percent': [r['battery_percent'] for r in readings],
    }
    return jsonify(data)


@app.route('/api/refresh')
def api_refresh():
    """Manually trigger a Victron reading refresh."""
    try:
        import asyncio
        data = asyncio.run(read_mppt_data())
        save_reading(data)
        # Convert enum values to strings for JSON serialization
        clean = {k: str(v) if hasattr(v, 'name') else v for k, v in data.items()}
        return jsonify({'success': True, 'data': clean})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/refresh/system')
def api_refresh_system():
    """Manually trigger a system data refresh."""
    try:
        data = collect_system_data()
        save_system_reading(data)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == "__main__":
    init_db()
    scheduler.start()

    print("=" * 70)
    print("HELIOAI DASHBOARD")
    print("=" * 70)
    print("\nStarting web server...")
    print("Dashboard: http://localhost:5000")
    print("Press Ctrl+C to stop\n")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    finally:
        scheduler.stop()
