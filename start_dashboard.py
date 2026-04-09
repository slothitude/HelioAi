"""
HelioAi Dashboard Launcher

Starts the web dashboard with background solar + system monitoring.
Run this script to launch HelioAi.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skills.victron_dashboard import app, scheduler
from skills.victron_monitor import init_db


if __name__ == "__main__":
    # Initialize database
    init_db()

    # Start the monitor schedulers (Victron every 10 min, system every 2 min)
    scheduler.start()

    print("=" * 70)
    print("  _    _      _       ___  _    ___ ")
    print(" | |  | |    | |     / _ \| |  / _ \ ")
    print(" | |__| | ___| |__  | | | | |_| | | |")
    print(" |  __  |/ _ \ '_ \ | | | | __| | | |")
    print(" | |  | |  __/ | | || |_| | |_| |_| |")
    print(" |_|  |_|\___|_| |_(_)___/ \__|\___/ ")
    print()
    print("  Solar Dashboard by SlothitudeGames")
    print("=" * 70)
    print("\n  Dashboard: http://localhost:5000")
    print("  Tailscale: http://100.84.161.63:5000")
    print("  Press Ctrl+C to stop\n")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    finally:
        scheduler.stop()
        print("\nHelioAi stopped.")
