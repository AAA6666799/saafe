# Manual Upload of Fire Detection AI Notebooks to AWS S3

This guide provides step-by-step instructions for manually uploading the Fire Detection AI training notebooks to AWS S3 using the AWS Management Console.

## Prerequisites

- AWS account with appropriate permissions
- Access to the AWS Management Console
- The Fire Detection AI notebooks on your local machine

## Step 1: Sign in to the AWS Management Console

1. Go to [https://aws.amazon.com/console/](https://aws.amazon.com/console/)
2. Sign in with your AWS account credentials

## Step 2: Navigate to S3

1. In the AWS Management Console, search for "S3" in the search bar at the top
2. Click on "S3" in the search results to open the S3 console

## Step 3: Create a Bucket (if needed)

If you don't already have a bucket for your notebooks:

1. Click the "Create bucket" button
2. Enter a unique bucket name (e.g., "fire-detection-training-notebooks")
3. Select the AWS Region where you plan to run SageMaker
4. Configure bucket settings:
   - For training data, you can leave the default settings
   - For production data, consider enabling versioning and encryption
5. Click "Create bucket"

## Step 4: Create Folders (Optional)

To organize your notebooks:

1. Navigate to your bucket
2. Click "Create folder"
3. Name the folder (e.g., "fire-detection-notebooks")
4. Click "Create folder"

## Step 5: Upload Notebooks

1. Navigate to your bucket or the folder you created
2. Click the "Upload" button
3. Click "Add files" or drag and drop files
4. Select the following notebooks from your local machine:
   - `fire_detection_5m_training_simplified.ipynb`
   - `fire_detection_5m_training_part1.ipynb`
   - `fire_detection_5m_training_part2.ipynb`
   - `fire_detection_5m_training_part3.ipynb`
   - `fire_detection_5m_training_part4.ipynb`
   - `fire_detection_5m_training_part5.ipynb`
   - `fire_detection_5m_training_part6.ipynb`
5. Click "Upload"

## Step 6: Upload Supporting Files

1. Navigate to your bucket or create a "supporting_files" folder
2. Click the "Upload" button
3. Click "Add files" or drag and drop files
4. Select the following supporting files:
   - `requirements_gpu.txt`
   - `fire_detection_50m_config.json`
   - `minimal_fire_ensemble_config.json`
5. Click "Upload"

## Step 7: Verify Uploads

1. Navigate through your bucket to ensure all files were uploaded correctly
2. Check that the file sizes match the original files
3. Note the S3 URI for your bucket (e.g., `s3://fire-detection-training-notebooks/`)

## Step 8: Set Permissions (Optional)

If you need to share these notebooks with others:

1. Select a file or folder
2. Click the "Actions" button
3. Select "Make public" or configure specific permissions
4. Follow the prompts to confirm

## Next Steps

Now that your notebooks are uploaded to S3, you can:

1. Launch a SageMaker notebook instance with ml.p3.16xlarge
2. Download the notebooks from S3 to your SageMaker instance
3. Run the notebooks on SageMaker

For detailed instructions on these next steps, refer to the `sagemaker_p3_16xlarge_guide.md` file.

## Downloading Notebooks to SageMaker

Once your SageMaker instance is running:

1. Open JupyterLab in your SageMaker instance
2. Click on the "+" button in the file browser to create a new launcher
3. Open a terminal
4. Use the AWS CLI to download your notebooks:

```bash
# Create a directory for your notebooks
mkdir -p fire_detection

# Download notebooks from S3
aws s3 cp s3://your-bucket-name/fire-detection-notebooks/ fire_detection/ --recursive

# Verify the download
ls -la fire_detection/
```

Replace `your-bucket-name` with your actual bucket name.

## Troubleshooting

- **Access Denied**: Check your IAM permissions for S3
- **Slow Uploads**: Consider using the AWS CLI for large files
- **File Corruption**: Verify file integrity after upload
- **Bucket Not Found**: Ensure the bucket name is correct and the bucket exists in your region