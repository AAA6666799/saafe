# 🔧 Troubleshooting JSON/YAML Errors in AWS Deployment

## Problem

When deploying to AWS, you may encounter JSON or YAML parsing errors that look like this:

```
Error: Failed to parse JSON/YAML
Error: Invalid character in JSON
Error: Unexpected token in JSON
```

## Root Cause

These errors are typically caused by hidden macOS metadata files that are automatically created when you extract or manipulate files on macOS. These files include:

- Files starting with `._` (e.g., `._eb-config.json`)
- `.DS_Store` files
- `__MACOSX` directories

When these files are included in your deployment package, they interfere with AWS's JSON/YAML parsing processes.

## Solutions

### Solution 1: Use Our Clean Deployment Script (Recommended)

We've provided a script that automatically creates a clean deployment package without hidden files:

```bash
python3 create_clean_deployment.py
```

This will create a file called `saafe-dashboard-clean.zip` that contains only the necessary files.

### Solution 2: Manual Cleanup

If you need to manually clean up a deployment package, you can remove the hidden files:

```bash
# Remove hidden files from a zip archive
zip -d your-deployment-package.zip "__MACOSX/*" "*.DS_Store" "._*"
```

### Solution 3: Prevent Hidden Files During Compression

When creating zip files on macOS, use these commands to exclude hidden files:

```bash
# Create zip without hidden files
zip -r saafe-dashboard.zip . -x ".*" "__MACOSX/*" "*.DS_Store" "._*"
```

## Prevention

To prevent this issue in the future:

1. Always use our clean deployment script
2. When creating zip files manually, exclude hidden files
3. Check your deployment packages before uploading to AWS

## Verification

To verify that your deployment package is clean:

```bash
# List contents of your zip file
unzip -l your-deployment-package.zip | grep -E "(^\s*[0-9]|^\s*----)"

# Look for any files starting with . or _ characters that shouldn't be there
```

A clean deployment package should only contain your actual application files without any hidden metadata files.

## Need Help?

If you continue to experience issues:

1. Run our clean deployment script
2. Check that all JSON/YAML configuration files are valid:
   ```bash
   # Validate JSON files
   python3 -m json.tool eb-config.json
   
   # Validate YAML files (if you have any)
   # You might need to install pyyaml first: pip install pyyaml
   python3 -c "import yaml; yaml.safe_load(open('your-file.yml'))"
   ```

3. Contact support with details about the error message you're seeing