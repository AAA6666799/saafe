#!/usr/bin/env python3
"""
AWS CLI Login Access Checker

This script checks if AWS CLI is properly configured and provides guidance
for setting up authentication if needed.
"""

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import subprocess
import sys

def check_aws_cli_configured():
    """
    Check if AWS CLI is configured by attempting to get caller identity.
    Returns True if configured, False otherwise.
    """
    try:
        # Try to create STS client and get caller identity
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()

        print("✅ AWS CLI is configured and accessible!")
        print(f"   Account ID: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")
        print(f"   Region: {boto3.Session().region_name}")
        return True

    except NoCredentialsError:
        print("❌ AWS credentials not found.")
        print("   You need to configure AWS CLI credentials.")
        return False

    except PartialCredentialsError:
        print("❌ Incomplete AWS credentials found.")
        print("   Your AWS credentials are incomplete (missing access key or secret key).")
        return False

    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidAccessKeyId':
            print("❌ Invalid AWS Access Key ID.")
            return False
        elif e.response['Error']['Code'] == 'SignatureDoesNotMatch':
            print("❌ Invalid AWS Secret Access Key.")
            return False
        else:
            print(f"❌ AWS Client Error: {e}")
            return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def setup_aws_configure():
    """
    Guide the user through AWS CLI configuration.
    """
    print("\n🔧 To configure AWS CLI, run the following command:")
    print("   aws configure")
    print("\n   You'll be prompted to enter:")
    print("   - AWS Access Key ID")
    print("   - AWS Secret Access Key")
    print("   - Default region name (e.g., us-east-1)")
    print("   - Default output format (json recommended)")

    print("\n💡 Alternative: Set environment variables:")
    print("   export AWS_ACCESS_KEY_ID='your-access-key'")
    print("   export AWS_SECRET_ACCESS_KEY='your-secret-key'")
    print("   export AWS_DEFAULT_REGION='us-east-1'")

    print("\n🔐 For production, consider using AWS IAM roles or AWS SSO.")

def check_aws_cli_installed():
    """
    Check if AWS CLI is installed.
    """
    try:
        result = subprocess.run(['aws', '--version'],
                              capture_output=True, text=True, check=True)
        print(f"✅ AWS CLI installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ AWS CLI not installed.")
        print("   Install it from: https://aws.amazon.com/cli/")
        return False

def main():
    print("🔍 Checking AWS CLI access...\n")

    # Check if AWS CLI is installed
    if not check_aws_cli_installed():
        return

    # Check if configured
    if not check_aws_cli_configured():
        setup_aws_configure()
        print("\n🔄 After configuration, run this script again to verify.")
        sys.exit(1)

    print("\n🎉 AWS CLI is ready to use!")

if __name__ == "__main__":
    main()