#!/usr/bin/env python3
"""
Saafe Fire Detection System - Cloud Deployment Entry Point
"""

import sys
from pathlib import Path

# Add deployment directory to Python path
deployment_dir = Path(__file__).parent / "deployment"
sys.path.insert(0, str(deployment_dir))

if __name__ == "__main__":
    try:
        from deploy import main
        sys.exit(main())
    except ImportError as e:
        print(f"❌ Failed to import deployment manager: {e}")
        print("Please ensure you're running this script from the correct directory.")
        sys.exit(1)