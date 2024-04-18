import shutil
import os
from urllib.request import urlretrieve
from urllib.parse import urlparse

class CleanData:
    def __init__(self, data_dir, urls):
        self.data_dir = data_dir
        self.urls = urls

        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    def sanitize_filename(self, url_or_path):
        """
        Extracts and sanitizes a filename from a URL or a local file path.
        """
        if url_or_path.startswith("http"):
            parsed_url = urlparse(url_or_path)
            filename = os.path.basename(parsed_url.path)
        else:
            filename = os.path.basename(url_or_path)
        
        for char in ['?', '&', '\\', '/', ':']:
            filename = filename.replace(char, '_')
        return filename

    def download_files(self, save_directory="test"):
        for url_or_path in self.urls:
            sanitized_filename = self.sanitize_filename(url_or_path)
            file_path = os.path.join(save_directory, sanitized_filename)

            # Ensure the save directory exists
            os.makedirs(save_directory, exist_ok=True)

            if url_or_path.startswith("http"):
                # Download the file from URL
                try:
                    urlretrieve(url_or_path, file_path)
                    print(f"Downloaded {url_or_path} to {file_path}")
                except Exception as e:
                    print(f"Error downloading {url_or_path}: {e}")
            else:
                # Handle local file
                if os.path.exists(url_or_path):
                    # Optionally, copy the file to the save_directory
                    shutil.copy2(url_or_path, file_path)
                    print(f"Copied local file {url_or_path} to {file_path}")
                else:
                    print(f"Local file {url_or_path} does not exist.")
