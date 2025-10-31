#!/usr/bin/env python3
"""
Launcher script for the Employee Promotion Prediction App
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app"""
    app_dir = Path(__file__).parent
    app_file = app_dir / "app.py"
    
    if not app_file.exists():
        print(f"Error: {app_file} not found!")
        sys.exit(1)
    
    print("🚀 Starting Employee Promotion Prediction App...")
    print("📊 Dashboard: View KPIs and predictions")
    print("🔮 Predictions: Upload data or use form input")
    print("📈 Model Analysis: Detailed model evaluation")
    print("🤖 AI Insights: AI-powered recommendations")
    print("\n" + "="*50)
    
    try:
        # Change to app directory
        os.chdir(app_dir)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

