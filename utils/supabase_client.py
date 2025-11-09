"""
Supabase Client for AURA
Handles file storage and database operations with Supabase
"""

from typing import Optional, List, Dict, Any, BinaryIO
from pathlib import Path
import io
from supabase import create_client, Client
from config.settings import settings
from utils.logger import logger

class SupabaseClient:
    """Manage Supabase storage and database operations"""
    
    def __init__(self):
        """Initialize Supabase connection"""
        self.url = settings.supabase_url
        self.key = settings.supabase_key
        self.bucket_name = settings.get('supabase.bucket_name', 'aura-documents')
        
        logger.info("Initializing Supabase connection...")
        try:
            self.client: Client = create_client(self.url, self.key)
            logger.info("Successfully connected to Supabase")
            
            # Verify bucket exists
            self._verify_bucket()
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {str(e)}")
            raise
    
    def _verify_bucket(self):
        """Verify that the storage bucket exists"""
        try:
            buckets = self.client.storage.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]
            
            if self.bucket_name in bucket_names:
                logger.info(f"Storage bucket '{self.bucket_name}' verified")
            else:
                logger.warning(f"Bucket '{self.bucket_name}' not found in: {bucket_names}")
                
        except Exception as e:
            logger.error(f"Error verifying bucket: {str(e)}")
    
    def upload_file(self, file_path: str, destination_path: str, content_type: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to Supabase storage
        
        Args:
            file_path: Local file path to upload
            destination_path: Destination path in bucket
            content_type: Optional MIME type
            
        Returns:
            Public URL of uploaded file or None if failed
        """
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Upload file
            response = self.client.storage.from_(self.bucket_name).upload(
                path=destination_path,
                file=file_data,
                file_options={"content-type": content_type} if content_type else None
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(destination_path)
            
            logger.info(f"Successfully uploaded file: {destination_path}")
            return public_url
            
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {str(e)}")
            return None
    
    def upload_bytes(self, file_data: bytes, destination_path: str, content_type: Optional[str] = None) -> Optional[str]:
        """
        Upload bytes data to Supabase storage
        
        Args:
            file_data: File data as bytes
            destination_path: Destination path in bucket
            content_type: Optional MIME type
            
        Returns:
            Public URL of uploaded file or None if failed
        """
        try:
            # Upload file
            response = self.client.storage.from_(self.bucket_name).upload(
                path=destination_path,
                file=file_data,
                file_options={"content-type": content_type} if content_type else None
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(destination_path)
            
            logger.info(f"Successfully uploaded bytes to: {destination_path}")
            return public_url
            
        except Exception as e:
            logger.error(f"Error uploading bytes to {destination_path}: {str(e)}")
            return None
    
    def download_file(self, file_path: str, local_path: str) -> bool:
        """
        Download a file from Supabase storage
        
        Args:
            file_path: Path in bucket
            local_path: Local path to save file
            
        Returns:
            True if successful
        """
        try:
            # Download file
            response = self.client.storage.from_(self.bucket_name).download(file_path)
            
            # Save to local file
            with open(local_path, 'wb') as f:
                f.write(response)
            
            logger.info(f"Successfully downloaded file: {file_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file {file_path}: {str(e)}")
            return False
    
    def download_bytes(self, file_path: str) -> Optional[bytes]:
        """
        Download file as bytes from Supabase storage
        
        Args:
            file_path: Path in bucket
            
        Returns:
            File data as bytes or None if failed
        """
        try:
            response = self.client.storage.from_(self.bucket_name).download(file_path)
            logger.info(f"Successfully downloaded bytes from: {file_path}")
            return response
            
        except Exception as e:
            logger.error(f"Error downloading bytes from {file_path}: {str(e)}")
            return None
    
    def list_files(self, path: str = "") -> List[Dict[str, Any]]:
        """
        List files in a bucket path
        
        Args:
            path: Path in bucket (empty for root)
            
        Returns:
            List of file metadata
        """
        try:
            files = self.client.storage.from_(self.bucket_name).list(path)
            logger.info(f"Listed {len(files)} files in path: {path}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {path}: {str(e)}")
            return []
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from Supabase storage
        
        Args:
            file_path: Path in bucket
            
        Returns:
            True if successful
        """
        try:
            self.client.storage.from_(self.bucket_name).remove([file_path])
            logger.info(f"Successfully deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def get_public_url(self, file_path: str) -> str:
        """
        Get public URL for a file
        
        Args:
            file_path: Path in bucket
            
        Returns:
            Public URL
        """
        return self.client.storage.from_(self.bucket_name).get_public_url(file_path)
    
    # Database operations
    def insert_record(self, table: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Insert a record into a database table
        
        Args:
            table: Table name
            data: Record data
            
        Returns:
            Inserted record or None
        """
        try:
            response = self.client.table(table).insert(data).execute()
            logger.info(f"Inserted record into {table}")
            return response.data[0] if response.data else None
            
        except Exception as e:
            logger.error(f"Error inserting record into {table}: {str(e)}")
            return None
    
    def query_records(self, table: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query records from a database table
        
        Args:
            table: Table name
            filters: Optional filters
            
        Returns:
            List of matching records
        """
        try:
            query = self.client.table(table).select("*")
            
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            response = query.execute()
            logger.info(f"Queried {len(response.data)} records from {table}")
            return response.data
            
        except Exception as e:
            logger.error(f"Error querying {table}: {str(e)}")
            return []
    
    def update_record(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        """
        Update a record in a database table
        
        Args:
            table: Table name
            record_id: Record ID
            data: Updated data
            
        Returns:
            True if successful
        """
        try:
            self.client.table(table).update(data).eq("id", record_id).execute()
            logger.info(f"Updated record {record_id} in {table}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating record in {table}: {str(e)}")
            return False
    
    def delete_record(self, table: str, record_id: str) -> bool:
        """
        Delete a record from a database table
        
        Args:
            table: Table name
            record_id: Record ID
            
        Returns:
            True if successful
        """
        try:
            self.client.table(table).delete().eq("id", record_id).execute()
            logger.info(f"Deleted record {record_id} from {table}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting record from {table}: {str(e)}")
            return False

# Global instance
supabase_client = SupabaseClient()
