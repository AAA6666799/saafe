#!/usr/bin/env python3
"""
System Status Checker
This script checks the status of all deployed components in the Synthetic Fire Prediction System.
"""

import boto3
import json

def check_lambda_functions():
    """Check the status of deployed Lambda functions"""
    print("🔍 Checking Lambda Functions...")
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    functions = [
        'saafe-monitoring-agent',
        'saafe-analysis-agent', 
        'saafe-response-agent'
    ]
    
    status = {}
    
    for function_name in functions:
        try:
            response = lambda_client.get_function(FunctionName=function_name)
            state = response['Configuration']['State']
            last_modified = response['Configuration']['LastModified']
            status[function_name] = {
                'state': state,
                'last_modified': last_modified
            }
            print(f"   ✅ {function_name}: {state}")
        except Exception as e:
            print(f"   ❌ {function_name}: ERROR - {e}")
            status[function_name] = {
                'state': 'ERROR',
                'error': str(e)
            }
    
    return status

def check_iam_role():
    """Check if the required IAM role exists"""
    print("\n🔍 Checking IAM Role...")
    
    iam_client = boto3.client('iam')
    
    try:
        response = iam_client.get_role(RoleName='SaafeLambdaExecutionRole')
        print("   ✅ SaafeLambdaExecutionRole: EXISTS")
        return True
    except Exception as e:
        print(f"   ❌ SaafeLambdaExecutionRole: NOT FOUND - {e}")
        return False

def check_cloudwatch_rules():
    """Check CloudWatch event rules"""
    print("\n🔍 Checking CloudWatch Rules...")
    
    events_client = boto3.client('events', region_name='us-east-1')
    
    try:
        response = events_client.describe_rule(Name='saafe-monitoring-schedule')
        print("   ✅ saafe-monitoring-schedule: EXISTS")
        return True
    except Exception as e:
        print(f"   ❌ saafe-monitoring-schedule: NOT FOUND - {e}")
        return False

def main():
    """Main function to check system status"""
    print("🔥 Synthetic Fire Prediction System - Status Check")
    print("=" * 52)
    
    # Check components
    lambda_status = check_lambda_functions()
    iam_status = check_iam_role()
    cw_status = check_cloudwatch_rules()
    
    # Summary
    print("\n" + "=" * 52)
    print("SYSTEM STATUS SUMMARY")
    print("=" * 52)
    
    lambda_ready = all(status['state'] == 'Active' for status in lambda_status.values())
    print(f"Lambda Functions: {'✅ READY' if lambda_ready else '❌ ISSUE'}")
    print(f"IAM Role: {'✅ READY' if iam_status else '❌ ISSUE'}")
    print(f"CloudWatch Rules: {'✅ READY' if cw_status else '❌ ISSUE'}")
    
    overall_status = lambda_ready and iam_status and cw_status
    print("\n" + "=" * 52)
    print(f"OVERALL SYSTEM STATUS: {'✅ OPERATIONAL' if overall_status else '❌ ISSUE'}")
    print("=" * 52)
    
    if overall_status:
        print("\n🎉 The Synthetic Fire Prediction System is fully operational!")
        print("   All components are deployed and ready for production use.")
    else:
        print("\n⚠️  The system has some issues that need to be addressed.")
        print("   Please check the individual component statuses above.")

if __name__ == "__main__":
    main()