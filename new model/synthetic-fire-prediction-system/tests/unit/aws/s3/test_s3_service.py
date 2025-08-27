"""
Unit tests for the S3ServiceImpl class.
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import boto3
import botocore
from io import BytesIO

from src.aws.s3.service import S3ServiceImpl


class TestS3ServiceImpl(unittest.TestCase):
    """Test cases for the S3ServiceImpl class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.config = {
            'default_bucket': 'test-bucket',
            'region_name': 'us-west-2',
            'aws_access_key_id': 'test-key',
            'aws_secret_access_key': 'test-secret'
        }
        
        # Create mock boto3 session and clients
        self.mock_session = MagicMock()
        self.mock_client = MagicMock()
        self.mock_resource = MagicMock()
        
        # Set up the mock session to return mock client and resource
        self.mock_session.client.return_value = self.mock_client
        self.mock_session.resource.return_value = self.mock_resource
        
        # Patch boto3.Session to return our mock session
        self.session_patcher = patch('boto3.Session', return_value=self.mock_session)
        self.mock_boto3_session = self.session_patcher.start()
        
        # Create S3 service with mocked session
        self.s3_service = S3ServiceImpl(self.config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patcher
        self.session_patcher.stop()
    
    def test_initialization(self):
        """Test initialization of S3ServiceImpl."""
        self.assertIsNotNone(self.s3_service)
        self.assertEqual(self.s3_service.config, self.config)
        
        # Check that session was initialized with correct parameters
        self.mock_boto3_session.assert_called_once_with(
            profile_name=None,
            region_name='us-west-2',
            aws_access_key_id='test-key',
            aws_secret_access_key='test-secret',
            aws_session_token=None
        )
        
        # Check that client and resource were initialized
        self.mock_session.client.assert_called_once_with('s3')
        self.mock_session.resource.assert_called_once_with('s3')
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        # Should not raise an exception
        self.s3_service.validate_config()
    
    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        # Missing default_bucket
        invalid_config = {
            'region_name': 'us-west-2'
        }
        
        with self.assertRaises(ValueError):
            S3ServiceImpl(invalid_config)
    
    def test_test_connection_success(self):
        """Test connection test with successful connection."""
        # Set up mock to return success
        self.mock_client.list_buckets.return_value = {'Buckets': []}
        
        # Test connection
        result = self.s3_service.test_connection()
        
        # Check result
        self.assertTrue(result)
        self.mock_client.list_buckets.assert_called_once()
    
    def test_test_connection_failure(self):
        """Test connection test with failed connection."""
        # Set up mock to raise exception
        self.mock_client.list_buckets.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
            'ListBuckets'
        )
        
        # Test connection
        result = self.s3_service.test_connection()
        
        # Check result
        self.assertFalse(result)
        self.mock_client.list_buckets.assert_called_once()
    
    @patch('os.path.exists', return_value=True)
    def test_upload_file_success(self, mock_exists):
        """Test uploading a file with success."""
        # Set up test parameters
        local_path = '/path/to/file.txt'
        s3_key = 'test/file.txt'
        
        # Set up mock to return success
        self.mock_client.upload_file.return_value = None
        
        # Upload file
        result = self.s3_service.upload_file(local_path, s3_key)
        
        # Check result
        self.assertTrue(result)
        self.mock_client.upload_file.assert_called_once_with(
            local_path, self.config['default_bucket'], s3_key
        )
    
    @patch('os.path.exists', return_value=True)
    def test_upload_file_with_bucket(self, mock_exists):
        """Test uploading a file with custom bucket."""
        # Set up test parameters
        local_path = '/path/to/file.txt'
        s3_key = 'test/file.txt'
        bucket = 'custom-bucket'
        
        # Set up mock to return success
        self.mock_client.upload_file.return_value = None
        
        # Upload file
        result = self.s3_service.upload_file(local_path, s3_key, bucket)
        
        # Check result
        self.assertTrue(result)
        self.mock_client.upload_file.assert_called_once_with(
            local_path, bucket, s3_key
        )
    
    @patch('os.path.exists', return_value=True)
    def test_upload_file_with_extra_args(self, mock_exists):
        """Test uploading a file with extra arguments."""
        # Set up test parameters
        local_path = '/path/to/file.txt'
        s3_key = 'test/file.txt'
        extra_args = {'ContentType': 'text/plain'}
        
        # Set up mock to return success
        self.mock_client.upload_file.return_value = None
        
        # Upload file
        result = self.s3_service.upload_file(local_path, s3_key, extra_args=extra_args)
        
        # Check result
        self.assertTrue(result)
        self.mock_client.upload_file.assert_called_once_with(
            local_path, self.config['default_bucket'], s3_key, ExtraArgs=extra_args
        )
    
    @patch('os.path.exists', return_value=False)
    def test_upload_file_not_found(self, mock_exists):
        """Test uploading a file that doesn't exist."""
        # Set up test parameters
        local_path = '/path/to/nonexistent.txt'
        s3_key = 'test/file.txt'
        
        # Upload file should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.s3_service.upload_file(local_path, s3_key)
        
        # Check that upload_file was not called
        self.mock_client.upload_file.assert_not_called()
    
    @patch('os.path.exists', return_value=True)
    def test_upload_file_client_error(self, mock_exists):
        """Test uploading a file with client error."""
        # Set up test parameters
        local_path = '/path/to/file.txt'
        s3_key = 'test/file.txt'
        
        # Set up mock to raise exception
        self.mock_client.upload_file.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
            'PutObject'
        )
        
        # Upload file
        result = self.s3_service.upload_file(local_path, s3_key)
        
        # Check result
        self.assertFalse(result)
        self.mock_client.upload_file.assert_called_once()
    
    @patch('os.makedirs')
    def test_download_file_success(self, mock_makedirs):
        """Test downloading a file with success."""
        # Set up test parameters
        s3_key = 'test/file.txt'
        local_path = '/path/to/file.txt'
        
        # Set up mock to return success
        self.mock_client.download_file.return_value = None
        
        # Download file
        result = self.s3_service.download_file(s3_key, local_path)
        
        # Check result
        self.assertTrue(result)
        mock_makedirs.assert_called_once()
        self.mock_client.download_file.assert_called_once_with(
            self.config['default_bucket'], s3_key, local_path
        )
    
    @patch('os.makedirs')
    def test_download_file_with_bucket(self, mock_makedirs):
        """Test downloading a file with custom bucket."""
        # Set up test parameters
        s3_key = 'test/file.txt'
        local_path = '/path/to/file.txt'
        bucket = 'custom-bucket'
        
        # Set up mock to return success
        self.mock_client.download_file.return_value = None
        
        # Download file
        result = self.s3_service.download_file(s3_key, local_path, bucket)
        
        # Check result
        self.assertTrue(result)
        mock_makedirs.assert_called_once()
        self.mock_client.download_file.assert_called_once_with(
            bucket, s3_key, local_path
        )
    
    @patch('os.makedirs')
    def test_download_file_client_error(self, mock_makedirs):
        """Test downloading a file with client error."""
        # Set up test parameters
        s3_key = 'test/file.txt'
        local_path = '/path/to/file.txt'
        
        # Set up mock to raise exception
        self.mock_client.download_file.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'NoSuchKey', 'Message': 'The specified key does not exist.'}},
            'GetObject'
        )
        
        # Download file
        result = self.s3_service.download_file(s3_key, local_path)
        
        # Check result
        self.assertFalse(result)
        mock_makedirs.assert_called_once()
        self.mock_client.download_file.assert_called_once()
    
    def test_list_objects_success(self):
        """Test listing objects with success."""
        # Set up test parameters
        prefix = 'test/'
        
        # Set up mock to return objects
        self.mock_client.list_objects_v2.return_value = {
            'Contents': [
                {
                    'Key': 'test/file1.txt',
                    'Size': 100,
                    'LastModified': '2023-01-01T00:00:00Z',
                    'ETag': '"abc123"'
                },
                {
                    'Key': 'test/file2.txt',
                    'Size': 200,
                    'LastModified': '2023-01-02T00:00:00Z',
                    'ETag': '"def456"'
                }
            ]
        }
        
        # List objects
        result = self.s3_service.list_objects(prefix)
        
        # Check result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['key'], 'test/file1.txt')
        self.assertEqual(result[0]['size'], 100)
        self.assertEqual(result[0]['etag'], 'abc123')
        self.assertEqual(result[1]['key'], 'test/file2.txt')
        self.assertEqual(result[1]['size'], 200)
        self.assertEqual(result[1]['etag'], 'def456')
        
        self.mock_client.list_objects_v2.assert_called_once_with(
            Bucket=self.config['default_bucket'], Prefix=prefix
        )
    
    def test_list_objects_empty(self):
        """Test listing objects with no results."""
        # Set up test parameters
        prefix = 'test/'
        
        # Set up mock to return no objects
        self.mock_client.list_objects_v2.return_value = {}
        
        # List objects
        result = self.s3_service.list_objects(prefix)
        
        # Check result
        self.assertEqual(result, [])
        self.mock_client.list_objects_v2.assert_called_once()
    
    def test_list_objects_with_bucket(self):
        """Test listing objects with custom bucket."""
        # Set up test parameters
        prefix = 'test/'
        bucket = 'custom-bucket'
        
        # Set up mock to return objects
        self.mock_client.list_objects_v2.return_value = {
            'Contents': [
                {
                    'Key': 'test/file1.txt',
                    'Size': 100,
                    'LastModified': '2023-01-01T00:00:00Z',
                    'ETag': '"abc123"'
                }
            ]
        }
        
        # List objects
        result = self.s3_service.list_objects(prefix, bucket)
        
        # Check result
        self.assertEqual(len(result), 1)
        self.mock_client.list_objects_v2.assert_called_once_with(
            Bucket=bucket, Prefix=prefix
        )
    
    def test_list_objects_client_error(self):
        """Test listing objects with client error."""
        # Set up test parameters
        prefix = 'test/'
        
        # Set up mock to raise exception
        self.mock_client.list_objects_v2.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
            'ListObjectsV2'
        )
        
        # List objects
        result = self.s3_service.list_objects(prefix)
        
        # Check result
        self.assertEqual(result, [])
        self.mock_client.list_objects_v2.assert_called_once()
    
    def test_upload_fileobj_success(self):
        """Test uploading a file-like object with success."""
        # Set up test parameters
        fileobj = BytesIO(b'test data')
        s3_key = 'test/file.txt'
        
        # Set up mock to return success
        self.mock_client.upload_fileobj.return_value = None
        
        # Upload file object
        result = self.s3_service.upload_fileobj(fileobj, s3_key)
        
        # Check result
        self.assertTrue(result)
        self.mock_client.upload_fileobj.assert_called_once_with(
            fileobj, self.config['default_bucket'], s3_key
        )
    
    def test_download_fileobj_success(self):
        """Test downloading to a file-like object with success."""
        # Set up test parameters
        fileobj = BytesIO()
        s3_key = 'test/file.txt'
        
        # Set up mock to return success
        self.mock_client.download_fileobj.return_value = None
        
        # Download to file object
        result = self.s3_service.download_fileobj(s3_key, fileobj)
        
        # Check result
        self.assertTrue(result)
        self.mock_client.download_fileobj.assert_called_once_with(
            self.config['default_bucket'], s3_key, fileobj
        )
    
    def test_delete_object_success(self):
        """Test deleting an object with success."""
        # Set up test parameters
        s3_key = 'test/file.txt'
        
        # Set up mock to return success
        self.mock_client.delete_object.return_value = {}
        
        # Delete object
        result = self.s3_service.delete_object(s3_key)
        
        # Check result
        self.assertTrue(result)
        self.mock_client.delete_object.assert_called_once_with(
            Bucket=self.config['default_bucket'], Key=s3_key
        )


if __name__ == '__main__':
    unittest.main()