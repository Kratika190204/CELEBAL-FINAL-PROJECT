#!/usr/bin/env python3
"""
Test script to verify layout fixes are working
"""

import os
import sys

def test_css_file():
    """Test that the CSS file contains the necessary layout fixes"""
    print("🧪 Testing CSS Layout Fixes...")
    
    try:
        with open('styles.css', 'r') as f:
            css_content = f.read()
        
        # Check for key CSS rules
        required_rules = [
            'max-width: 100vw',
            'overflow-x: hidden',
            'dataframe-container',
            'main-content',
            '!important'
        ]
        
        missing_rules = []
        for rule in required_rules:
            if rule not in css_content:
                missing_rules.append(rule)
        
        if missing_rules:
            print(f"   ❌ Missing CSS rules: {missing_rules}")
            return False
        else:
            print("   ✅ All required CSS rules found")
            return True
            
    except FileNotFoundError:
        print("   ❌ styles.css file not found")
        return False
    except Exception as e:
        print(f"   ❌ Error reading CSS file: {e}")
        return False

def test_app_structure():
    """Test that the app.py has proper container wrappers"""
    print("🧪 Testing App Structure...")
    
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # Check for container wrappers
        required_containers = [
            'main-content',
            'dataframe-container',
            'metric-grid',
            'chart-container'
        ]
        
        missing_containers = []
        for container in required_containers:
            if container not in app_content:
                missing_containers.append(container)
        
        if missing_containers:
            print(f"   ❌ Missing container wrappers: {missing_containers}")
            return False
        else:
            print("   ✅ All required container wrappers found")
            return True
            
    except FileNotFoundError:
        print("   ❌ app.py file not found")
        return False
    except Exception as e:
        print(f"   ❌ Error reading app.py: {e}")
        return False

def main():
    """Run all layout tests"""
    print("🔧 Testing Layout Fixes...\n")
    
    css_test = test_css_file()
    app_test = test_app_structure()
    
    if css_test and app_test:
        print("\n✅ All layout fixes are properly implemented!")
        print("\n🎯 To test the fixes:")
        print("1. Run: streamlit run app.py")
        print("2. Click 'Generate Sample Data'")
        print("3. Check that content doesn't expand to the right")
        print("4. Test on different screen sizes")
        return True
    else:
        print("\n❌ Some layout fixes are missing!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 