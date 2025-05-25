"""
Data Cleaning Module for KCC Chatbot

This module provides functionality to clean and preprocess the Kisan Call Centre (KCC) dataset.
It handles large CSV files by processing them in chunks to manage memory efficiently.
Only the first 20000 rows of data will be processed.

The main functionality includes:
- Loading data in manageable chunks
- Extracting relevant columns (QueryText and KccAns)
- Removing rows with missing values
- Converting text to lowercase
- Saving the cleaned data to a new CSV file
"""

import pandas as pd
import os
import logging

# Configure logging to track the data processing operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='data_cleaning.log', filemode='a')
logger = logging.getLogger(__name__)

class DataProcessing:
    """
    A class for processing and cleaning KCC dataset files.
    
    This class handles large CSV files by processing them in chunks,
    which helps to manage memory usage efficiently when dealing with
    large datasets.
    
    Attributes:
        file_path (str): Path to the input CSV file containing raw KCC data
        output_file (str): Path where the cleaned data will be saved
        file_exists (bool): Flag indicating if the output file already exists
    """

    def __init__(self, file_path, output_file):
        """
        Initialize the DataProcessing object.
        
        Args:
            file_path (str): Path to the input CSV file
            output_file (str): Path where the cleaned data will be saved
        """
        self.file_path = file_path
        self.output_file = output_file
        self.file_exists = os.path.isfile(output_file)
    
    def clean_data(self):
        """
        Clean and process the KCC dataset.
        
        This method:
        1. Reads the CSV file in chunks to manage memory
        2. Extracts only the 'QueryText' and 'KccAns' columns
        3. Removes rows with missing values
        4. Converts text to lowercase
        5. Saves the processed data to the output file
        6. For development purposes, only processes the first chunk (limited to 100,000 rows)
        
        Returns:
            None: The method outputs the cleaned data to the specified output file
        """
        # Set the chunk size (number of rows per chunk)
        chunk_size = 20000  # Process 20,000 rows at a time to manage memory usage
        
        logger.info(f"Starting data cleaning process from {self.file_path}")
        logger.info(f"Output will be saved to {self.output_file}")
        
        # Create an iterator for the file to process it in chunks
        chunk_iter = pd.read_csv(self.file_path, chunksize=chunk_size, low_memory=False)
        
        # Process each chunk of data
        for i, chunk in enumerate(chunk_iter):
            logger.info(f"Processing chunk {i+1}")
            
            # Extract only the relevant columns: query text and KCC answers
            data = chunk[['QueryText', 'KccAns']]
            
            # Remove rows with missing values in either column
            data = data.dropna()
            logger.info(f"After removing missing values: {len(data)} rows remaining")
            
            # Convert all text data to lowercase for consistency
            data = data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
            
            # Handle file creation or appending based on whether it's the first chunk
            if i == 0 and not self.file_exists:
                # First iteration and file doesn't exist: create with headers
                data.to_csv(self.output_file, index=False, mode='w')
                logger.info(f"Created new file: {self.output_file}")
                print(f"Created new file: {self.output_file}")
                break  # Limiting the dataset to only the first chunk (100,000 rows) for development
            else:
                # Append to existing file without writing headers again
                data.to_csv(self.output_file, index=False, mode='a', header=False)
                logger.info(f"Appended chunk {i+1} to {self.output_file}")
                print(f"Appended chunk {i+1} to {self.output_file}")
        
        logger.info("Data cleaning process completed successfully")
        print("Processing complete.")

if __name__ == '__main__':
    """
    Main execution block that runs when the script is executed directly.
    
    This will:
    1. Define input and output file paths
    2. Create a DataProcessing instance
    3. Execute the data cleaning process
    """
    # Path to the original KCC dataset CSV file
    file_path = 'kcc_dataset.csv'
    
    # Path where the cleaned dataset will be saved
    # Using a different name to avoid overwriting the original file
    output_file = 'new_cleaned_kcc_dataset.csv'
    
    logger.info("Starting data cleaning script")
    
    try:
        # Create an instance of DataProcessing with the specified file paths
        data_processor = DataProcessing(file_path, output_file)
        
        # Execute the data cleaning process
        data_processor.clean_data()
        
        logger.info("Data cleaning script completed successfully")
    except Exception as e:
        logger.error(f"Error in data cleaning process: {str(e)}")
        print(f"Error: {str(e)}")
