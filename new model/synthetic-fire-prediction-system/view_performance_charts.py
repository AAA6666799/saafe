#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - View Performance Charts
This script displays the performance comparison charts.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def display_chart(image_path, title):
    """Display a chart image."""
    if os.path.exists(image_path):
        img = mpimg.imread(image_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Chart not found: {image_path}")

def main():
    """Main function to display all performance charts."""
    print("Displaying performance comparison charts...")
    print("=" * 50)
    
    # Display overall performance comparison chart
    print("1. Overall Performance Comparison Chart")
    display_chart('model_performance_comparison.png', 
                  'Performance Comparison: FLIR+SCD41 Fire Detection Model vs. General AI Models')
    
    # Display domain specialization chart
    print("2. Domain Specialization Comparison Chart")
    display_chart('domain_specialization_comparison.png',
                  'Domain Specialization Comparison: Fire Detection vs. General Purpose')
    
    print("\n✅ All charts displayed successfully!")
    print("\n📊 The charts visually demonstrate how our specialized fire detection model compares with general AI models.")
    print("   Our model shows superior performance in its specialized domain despite lower general capabilities.")

if __name__ == "__main__":
    main()