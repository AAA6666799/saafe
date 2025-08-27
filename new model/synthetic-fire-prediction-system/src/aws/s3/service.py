"""
Amazon S3 service implementation.

This module provides concrete implementations of the S3Service interface
for storing and retrieving data from Amazon S3.
"""

from typing import Dict, Any, List, Optional, BinaryIO
import os
import boto3
import botocore
from datetime import datetime

from ..base import S3Service


class S3ServiceImpl(S3Service):
    """
    Implementation of the S3Service interface for Amazon S3 operations.
    
    This class provides concrete implementations of the abstract methods
    defined in the S3Service base class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the S3 service implementation.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.initialize_client()
        self.initialize_resource()
    
    def upload_file(self, 
                   local_path: str, 
                   s3_key: str, 
                   bucket: Optional[str] = None,
                   extra_args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
            bucket: Optional bucket name (uses default_bucket if not provided)
            extra_args: Optional extra arguments for upload
            
        Returns:
            True if upload is successful, False otherwise
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        bucket_name = bucket if bucket else self.config['default_bucket']
        
        try:
            if extra_args:
                self.client.upload_file(local_path, bucket_name, s3_key, ExtraArgs=extra_args)
            else:
                self.client.upload_file(local_path, bucket_name, s3_key)
            return True
        except botocore.exceptions.ClientError as e:
            print(f"Error uploading file to S3: {e}")
            return False
    
    def download_file(self, 
                     s3_key: str, 
                     local_path: str, 
                     bucket: Optional[str] = None) -> bool:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path
            bucket: Optional bucket name (uses default_bucket if not provided)
            
        Returns:
            True if download is successful, False otherwise
        """
        bucket_name = bucket if bucket else self.config['default_bucket']
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
        try:
            self.client.download_file(bucket_name, s3_key, local_path)
            return True
        except botocore.exceptions.ClientError as e:
            print(f"Error downloading file from S3: {e}")
            return False
    
    def list_objects(self, 
                    prefix: str, 
                    bucket: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket.
        
        Args:
            prefix: Prefix to filter objects
            bucket: Optional bucket name (uses default_bucket if not provided)
            
        Returns:
            List of object information dictionaries
        """
        bucket_name = bucket if bucket else self.config['default_bucket']
        
        try:
            response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            
            if 'Contents' not in response:
                return []
            
            return [
                {
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag'].strip('"')
                }
                for obj in response['Contents']
            ]
        except botocore.exceptions.ClientError as e:
            print(f"Error listing objects in S3: {e}")
            return []
    
    def upload_fileobj(self, 
                      fileobj: BinaryIO, 
                      s3_key: str, 
                      bucket: Optional[str] = None,
                      extra_args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upload a file-like object to S3.
        
        Args:
            fileobj: File-like object
            s3_key: S3 object key
            bucket: Optional bucket name (uses default_bucket if not provided)
            extra_args: Optional extra arguments for upload
            
        Returns:
            True if upload is successful, False otherwise
        """
        bucket_name = bucket if bucket else self.config['default_bucket']
        
        try:
            if extra_args:
                self.client.upload_fileobj(fileobj, bucket_name, s3_key, ExtraArgs=extra_args)
            else:
                self.client.upload_fileobj(fileobj, bucket_name, s3_key)
            return True
        except botocore.exceptions.ClientError as e:
            print(f"Error uploading file object to S3: {e}")
            return False
    
    def download_fileobj(self, 
                        s3_key: str, 
                        fileobj: BinaryIO, 
                        bucket: Optional[str] = None) -> bool:
        """
        Download a file from S3 into a file-like object.
        
        Args:
            s3_key: S3 object key
            fileobj: File-like object to write to
            bucket: Optional bucket name (uses default_bucket if not provided)
            
        Returns:
            True if download is successful, False otherwise
        """
        bucket_name = bucket if bucket else self.config['default_bucket']
        
        try:
            self.client.download_fileobj(bucket_name, s3_key, fileobj)
            return True
        except botocore.exceptions.ClientError as e:
            print(f"Error downloading file object from S3: {e}")
            return False
    
    def delete_object(self, 
                     s3_key: str, 
                     bucket: Optional[str] = None) -> bool:
        """
        Delete an object from S3.
        
        Args:
            s3_key: S3 object key
            bucket: Optional bucket name (uses default_bucket if not provided)
            
        Returns:
            True if deletion is successful, False otherwise
        """
        bucket_name = bucket if bucket else self.config['default_bucket']
        
        try:
            self.client.delete_object(Bucket=bucket_name, Key=s3_key)
            return True
        except botocore.exceptions.ClientError as e:
            print(f"Error deleting object from S3: {e}")
            return False