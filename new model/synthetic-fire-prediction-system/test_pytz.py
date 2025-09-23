#!/usr/bin/env python3
"""
Test pytz import
"""

try:
    import pytz
    from datetime import datetime
    
    print("✅ pytz import successful")
    
    # Test creating a UTC datetime
    utc_now = datetime.now(pytz.UTC)
    print(f"Current UTC time: {utc_now}")
    
    print("✅ All tests passed")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()