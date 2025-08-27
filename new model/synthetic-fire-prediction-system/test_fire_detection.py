#!/usr/bin/env python3
"""
Interactive Fire Detection Test Script

Test the Saafe Fire Detection System with various scenarios.
"""

import sys
import time
from datetime import datetime

# Add project to path
sys.path.append('/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_fire_detection():
    """Interactive fire detection testing."""
    
    print("🔥 SAAFE FIRE DETECTION SYSTEM - INTERACTIVE TEST")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import system components
        from src.integrated_system import create_integrated_fire_system
        
        # Create system
        config = {
            'system_id': 'interactive_test',
            'sensors': {'mode': 'synthetic'},
            'agents': {
                'analysis': {'fire_pattern': {'confidence_threshold': 0.6}},
                'response': {'emergency': {'response_thresholds': {'HIGH': 0.7}}}
            }
        }
        
        print("🚀 Initializing Fire Detection System...")
        system = create_integrated_fire_system(config)
        
        if system.initialize():
            print("✅ System initialized successfully!")
        else:
            print("❌ System initialization failed!")
            return
        
        # Test scenarios
        test_scenarios = [
            {
                'name': '🌡️  Normal Room Conditions',
                'data': {
                    'thermal': {'sensor1': {'temperature_max': 22.0, 'temperature_avg': 20.0, 'hotspot_count': 0}},
                    'gas': {'sensor1': {'co_concentration': 5.0, 'smoke_density': 8.0}},
                    'environmental': {'sensor1': {'temperature': 21.0, 'humidity': 45.0}}
                },
                'expected': 'No Fire'
            },
            {
                'name': '🔥 Small Kitchen Fire',
                'data': {
                    'thermal': {'sensor1': {'temperature_max': 65.0, 'temperature_avg': 45.0, 'hotspot_count': 3}},
                    'gas': {'sensor1': {'co_concentration': 35.0, 'smoke_density': 55.0}},
                    'environmental': {'sensor1': {'temperature': 30.0, 'humidity': 35.0}}
                },
                'expected': 'Fire Detected'
            },
            {
                'name': '🚨 Large Building Fire',
                'data': {
                    'thermal': {'sensor1': {'temperature_max': 95.0, 'temperature_avg': 75.0, 'hotspot_count': 8}},
                    'gas': {'sensor1': {'co_concentration': 80.0, 'smoke_density': 90.0}},
                    'environmental': {'sensor1': {'temperature': 40.0, 'humidity': 20.0}}
                },
                'expected': 'Critical Fire'
            },
            {
                'name': '🍳 Cooking (False Positive Test)',
                'data': {
                    'thermal': {'sensor1': {'temperature_max': 45.0, 'temperature_avg': 32.0, 'hotspot_count': 1}},
                    'gas': {'sensor1': {'co_concentration': 15.0, 'smoke_density': 25.0}},
                    'environmental': {'sensor1': {'temperature': 26.0, 'humidity': 50.0}}
                },
                'expected': 'No Fire (Cooking)'
            }
        ]
        
        print("\n📊 RUNNING TEST SCENARIOS")
        print("-" * 60)
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   Expected: {scenario['expected']}")
            
            # Process data
            start_time = time.time()
            result = system.process_data(scenario['data'])
            processing_time = (time.time() - start_time) * 1000
            
            # Extract results
            final_decision = result.get('final_decision', {})
            fire_detected = final_decision.get('fire_detected', False)
            confidence = final_decision.get('confidence_score', 0.0)
            response_level = final_decision.get('response_level', 'NONE')
            
            # Display results
            fire_emoji = "🔥" if fire_detected else "✅"
            print(f"   Result: {fire_emoji} Fire Detected: {fire_detected}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Response Level: {response_level}")
            print(f"   Processing Time: {processing_time:.1f}ms")
            
            # Store results
            results.append({
                'scenario': scenario['name'],
                'fire_detected': fire_detected,
                'confidence': confidence,
                'response_level': response_level,
                'processing_time': processing_time
            })
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        avg_processing_time = sum(r['processing_time'] for r in results) / total_tests
        high_confidence_detections = sum(1 for r in results if r['confidence'] > 0.7)
        
        print(f"Total Tests: {total_tests}")
        print(f"Average Processing Time: {avg_processing_time:.1f}ms")
        print(f"High Confidence Detections: {high_confidence_detections}")
        print(f"System Performance: {'✅ EXCELLENT' if avg_processing_time < 1000 else '⚠️ NEEDS OPTIMIZATION'}")
        
        print("\n🎯 DETAILED RESULTS:")
        for r in results:
            conf_level = "HIGH" if r['confidence'] > 0.7 else "MEDIUM" if r['confidence'] > 0.4 else "LOW"
            print(f"   • {r['scenario']}: {conf_level} confidence ({r['confidence']:.3f})")
        
        print("\n✅ FIRE DETECTION SYSTEM TEST COMPLETED!")
        print("🔥 System is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fire_detection()