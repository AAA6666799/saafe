"""
Script to run the Saafe Fire Detection Dashboard
"""

import subprocess
import sys
import os


def main():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Saafe Fire Detection Dashboard...")
    print("=" * 50)
    
    try:
        # Change to the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)
        
        # Run the Streamlit dashboard on port 8505
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "dashboard.py", "--server.port", "8505"
        ]
        
        print("Starting dashboard on http://localhost:8505")
        print("Press Ctrl+C to stop the dashboard")
        print("=" * 50)
        
        # Run the command
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())