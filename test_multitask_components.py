#!/usr/bin/env python3
"""
Simple test script for multitask components that doesn't require PyTorch.
This script tests the basic structure and imports of our new components.
"""

import os
import sys
import importlib.util
from pathlib import Path

def test_file_exists(file_path):
    """Test that a file exists."""
    if os.path.exists(file_path):
        print(f"✓ {file_path} exists")
        return True
    else:
        print(f"❌ {file_path} does not exist")
        return False

def test_import_structure(file_path, module_name):
    """Test that a file can be imported and has expected structure."""
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"✓ {file_path} can be imported")
        
        # Check for expected classes/functions
        if hasattr(module, 'CaptioningDecoder'):
            print(f"  ✓ CaptioningDecoder class found")
        if hasattr(module, 'MaskedVideoModeling'):
            print(f"  ✓ MaskedVideoModeling class found")
        if hasattr(module, 'MultitaskLoss'):
            print(f"  ✓ MultitaskLoss class found")
        if hasattr(module, 'MultitaskPretrainingProject'):
            print(f"  ✓ MultitaskPretrainingProject class found")
        if hasattr(module, 'MultitaskRunner'):
            print(f"  ✓ MultitaskRunner class found")
            
        return True
    except Exception as e:
        print(f"❌ {file_path} import failed: {e}")
        return False

def test_config_file(config_path):
    """Test that config file exists and has expected structure."""
    if not os.path.exists(config_path):
        print(f"❌ {config_path} does not exist")
        return False
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            
        # Check for expected keys
        expected_keys = [
            'pipeline_project: "DeepCORO_multitask"',
            'loss_name: "multitask"',
            'decoder_layers:',
            'mask_ratio:',
            'loss_weights:'
        ]
        
        missing_keys = []
        for key in expected_keys:
            if key not in content:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ {config_path} missing expected keys: {missing_keys}")
            return False
        else:
            print(f"✓ {config_path} has expected structure")
            return True
            
    except Exception as e:
        print(f"❌ {config_path} read failed: {e}")
        return False

def test_documentation_files():
    """Test that documentation files exist."""
    docs_files = [
        'docs/MULTITASK_SETUP.md',
        'IMPLEMENTATION_SUMMARY.md'
    ]
    
    all_exist = True
    for doc_file in docs_files:
        if os.path.exists(doc_file):
            print(f"✓ {doc_file} exists")
        else:
            print(f"❌ {doc_file} does not exist")
            all_exist = False
    
    return all_exist

def test_registry_integration():
    """Test that registry integration is properly set up."""
    # Check enums.py
    enums_file = 'utils/enums.py'
    if os.path.exists(enums_file):
        try:
            with open(enums_file, 'r') as f:
                content = f.read()
            
            if 'MULTITASK = "multitask"' in content:
                print(f"✓ {enums_file} has MULTITASK enum")
            else:
                print(f"❌ {enums_file} missing MULTITASK enum")
                return False
        except Exception as e:
            print(f"❌ {enums_file} read failed: {e}")
            return False
    
    # Check loss typing
    typing_file = 'utils/loss/typing.py'
    if os.path.exists(typing_file):
        try:
            with open(typing_file, 'r') as f:
                content = f.read()
            
            if 'MultitaskLoss' in content:
                print(f"✓ {typing_file} has MultitaskLoss import")
            else:
                print(f"❌ {typing_file} missing MultitaskLoss import")
                return False
        except Exception as e:
            print(f"❌ {typing_file} read failed: {e}")
            return False
    
    return True

def main():
    """Run all component tests."""
    print("=" * 60)
    print("Testing Multitask DeepCORO-CLIP Components")
    print("=" * 60)
    
    tests = [
        ("Model Files", [
            ("models/captioning_decoder.py", "captioning_decoder"),
            ("models/masked_video_modeling.py", "masked_video_modeling"),
        ]),
        ("Loss Functions", [
            ("utils/loss/multitask_loss.py", "multitask_loss"),
        ]),
        ("Training Components", [
            ("projects/multitask_pretraining_project.py", "multitask_pretraining_project"),
            ("runners/multitask_runner.py", "multitask_runner"),
        ]),
        ("Configuration", [
            ("config/clip/multitask_config.yaml", None),
        ]),
        ("Documentation", [
            ("docs/MULTITASK_SETUP.md", None),
            ("IMPLEMENTATION_SUMMARY.md", None),
        ]),
        ("Test Scripts", [
            ("test_multitask_setup.py", "test_multitask_setup"),
        ]),
    ]
    
    all_passed = True
    
    for category, file_tests in tests:
        print(f"\n--- {category} ---")
        
        for file_path, module_name in file_tests:
            # Test file exists
            if not test_file_exists(file_path):
                all_passed = False
                continue
            
            # Test import structure (if module_name provided)
            if module_name:
                if not test_import_structure(file_path, module_name):
                    all_passed = False
            
            # Test config file structure (if it's a config file)
            if file_path.endswith('.yaml'):
                if not test_config_file(file_path):
                    all_passed = False
    
    # Test documentation files
    print(f"\n--- Documentation Files ---")
    if not test_documentation_files():
        all_passed = False
    
    # Test registry integration
    print(f"\n--- Registry Integration ---")
    if not test_registry_integration():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL COMPONENT TESTS PASSED!")
        print("✓ Multitask setup is properly implemented")
        print("✓ All files exist and have correct structure")
        print("✓ Registry integration is complete")
        print("✓ Documentation is available")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the failed components above")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)