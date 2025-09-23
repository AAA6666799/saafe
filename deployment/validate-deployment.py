#!/usr/bin/env python3
"""
Saafe Fire Detection System - Deployment Configuration Validator
This script validates that all deployment configurations are correctly set up.
"""

import os
import sys
from pathlib import Path
import yaml
import json

def validate_docker_compose(deployment_dir):
    """Validate Docker Compose configuration"""
    compose_file = deployment_dir / "docker-compose.yml"
    if not compose_file.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    try:
        with open(compose_file, 'r') as f:
            yaml.safe_load(f)
        print("✅ docker-compose.yml is valid YAML")
        return True
    except Exception as e:
        print(f"❌ docker-compose.yml validation failed: {e}")
        return False

def validate_kubernetes_config(deployment_dir):
    """Validate Kubernetes configuration"""
    kube_file = deployment_dir / "kubernetes-deployment.yaml"
    if not kube_file.exists():
        print("❌ kubernetes-deployment.yaml not found")
        return False
    
    try:
        with open(kube_file, 'r') as f:
            # Try to parse as YAML (multiple documents)
            documents = list(yaml.safe_load_all(f))
            if not documents:
                print("❌ kubernetes-deployment.yaml is empty")
                return False
        print("✅ kubernetes-deployment.yaml is valid YAML")
        return True
    except Exception as e:
        print(f"❌ kubernetes-deployment.yaml validation failed: {e}")
        return False

def validate_service_file(deployment_dir):
    """Validate systemd service file"""
    service_file = deployment_dir / "saafe-fire-detection.service"
    if not service_file.exists():
        print("❌ saafe-fire-detection.service not found")
        return False
    
    try:
        with open(service_file, 'r') as f:
            content = f.read()
            # Basic validation - check for required sections
            required_sections = ["[Unit]", "[Service]", "[Install]"]
            for section in required_sections:
                if section not in content:
                    print(f"❌ saafe-fire-detection.service missing section: {section}")
                    return False
        print("✅ saafe-fire-detection.service has required sections")
        return True
    except Exception as e:
        print(f"❌ saafe-fire-detection.service validation failed: {e}")
        return False

def validate_aws_script(deployment_dir):
    """Validate AWS deployment script"""
    deploy_script = deployment_dir / "deploy-aws.sh"
    if not deploy_script.exists():
        print("❌ deploy-aws.sh not found")
        return False
    
    try:
        # Check if script is executable
        if not os.access(deploy_script, os.X_OK):
            print("⚠️  deploy-aws.sh is not executable (will be fixed during deployment)")
        
        # Basic content validation
        with open(deploy_script, 'r') as f:
            content = f.read()
            # Check for key components
            required_elements = ["AWS_REGION", "ECS_CLUSTER_NAME", "ECR_REPOSITORY_NAME"]
            for element in required_elements:
                if element not in content:
                    print(f"❌ deploy-aws.sh missing element: {element}")
                    return False
        print("✅ deploy-aws.sh has required elements")
        return True
    except Exception as e:
        print(f"❌ deploy-aws.sh validation failed: {e}")
        return False

def validate_python_script(deployment_dir):
    """Validate Python deployment script"""
    deploy_script = deployment_dir / "deploy.py"
    if not deploy_script.exists():
        print("❌ deploy.py not found")
        return False
    
    try:
        # Try to import the script
        import importlib.util
        spec = importlib.util.spec_from_file_location("deploy", deploy_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ deploy.py is valid Python")
        return True
    except Exception as e:
        print(f"❌ deploy.py validation failed: {e}")
        return False

def main():
    print("🔍 Saafe Fire Detection System - Deployment Configuration Validator")
    print("=" * 70)
    
    # Determine project root
    current_dir = Path(__file__).parent
    project_root = current_dir if current_dir.name == "deployment" else current_dir / "deployment"
    
    if not project_root.exists():
        print("❌ Deployment directory not found")
        return 1
    
    print(f"📁 Using deployment directory: {project_root}")
    
    # Run all validations
    validations = [
        ("Docker Compose", lambda: validate_docker_compose(project_root)),
        ("Kubernetes Config", lambda: validate_kubernetes_config(project_root)),
        ("Systemd Service", lambda: validate_service_file(project_root)),
        ("AWS Script", lambda: validate_aws_script(project_root)),
        ("Python Script", lambda: validate_python_script(project_root))
    ]
    
    results = []
    for name, validator in validations:
        print(f"\n🧪 Validating {name}...")
        result = validator()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All deployment configurations are valid!")
        return 0
    else:
        print(f"\n⚠️  {failed} configuration(s) need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())