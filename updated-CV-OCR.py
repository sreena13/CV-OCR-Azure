import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import os
import pyodbc

# Azure Blob Storage Config
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=numberplates;AccountKey=UVB9HnEAWKDlq5lbGafLsz183mne7Y216RHliq4LxEyCflFo3SADIvALVmI1HSS8miXH2frLqshr+AStUDZCjg==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "vehicle-plate-images"

# Azure OCR Config
COMPUTER_VISION_ENDPOINT = "https://numberplate-cv.cognitiveservices.azure.com/"
COMPUTER_VISION_SUBSCRIPTION_KEY = "71kxuEKo6qhfr08KBcyuXmCk7BzetT7HoJkyz1NGqUk3Hg2kImPtJQQJ99ALACGhslBXJ3w3AAAFACOG1HuX"

# Azure SQL Database Config
server = 'serverocr.database.windows.net'
database = 'License-Plate-DB'
username = 'ggrserver'
password = 'Sreenamahesh@me'

# Initialize Clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
computer_vision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, CognitiveServicesCredentials(COMPUTER_VISION_SUBSCRIPTION_KEY))

def create_connection():
    """Establishes connection to the Azure SQL database."""
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def insert_license_plate_to_db(license_plate_text):
    """Inserts the detected license plate text into the database."""
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Assuming you have a table called OCRDetails with columns 'LicensePlate'
            for text in license_plate_text:
                query = "INSERT INTO OCRDetails (LicensePlate) VALUES (?)"
                cursor.execute(query, (text,))
            conn.commit()
            print("License plate text saved to the database successfully!")
        except Exception as e:
            print(f"Error inserting data into the database: {e}")
        finally:
            conn.close()
    else:
        print("Failed to connect to the database.")

def download_image_from_blob(blob_name, download_path):
    """Downloads an image from Azure Blob Storage."""
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(download_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        print(f"Image downloaded to {download_path}")
        return True
    except Exception as e:
        print(f"Failed to download image from Blob Storage: {e}")
        return False

def preprocess_image(image_path):
    """Preprocesses the image for better OCR detection."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read. Check the file path.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        enhanced_image_path = "enhanced_image.jpg"
        cv2.imwrite(enhanced_image_path, thresh)
        print(f"Preprocessed image saved at {enhanced_image_path}")
        return enhanced_image_path
    except Exception as e:
        print(f"Failed to preprocess image: {e}")
        return None

def perform_ocr(image_path):
    """Performs OCR to detect text in the image."""
    try:
        with open(image_path, "rb") as image_stream:
            result = computer_vision_client.read_in_stream(image_stream, raw=True)
            operation_location = result.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]

        # Poll for the OCR result
        while True:
            result = computer_vision_client.get_read_result(operation_id)
            if result.status not in [OperationStatusCodes.not_started, OperationStatusCodes.running]:
                break

        if result.status == OperationStatusCodes.succeeded:
            license_plate_text = []
            for read_result in result.analyze_result.read_results:
                for line in read_result.lines:
                    license_plate_text.append(line.text)
            return license_plate_text
    except Exception as e:
        print(f"Failed to perform OCR: {e}")
    return []

def main():
    """Main function to execute the OCR process and store data."""
    # Image details
    blob_name = "Cars5.png"
    local_image_path = "D:/PYTHON_ML/Data Sets/images/" + blob_name

    # Step 1: Check if the image is local or in Blob Storage
    if not os.path.exists(local_image_path):
        print(f"Image {local_image_path} not found locally. Attempting to download from Blob Storage...")
        if not download_image_from_blob(blob_name, local_image_path):
            print("Image could not be retrieved from Blob Storage. Exiting...")
            return

    # Step 2: Preprocess the image
    enhanced_image_path = preprocess_image(local_image_path)
    if not enhanced_image_path:
        print("Preprocessing failed. Exiting...")
        return

    # Step 3: Perform OCR
    detected_text = perform_ocr(enhanced_image_path)
    print("Detected License Plate Text:", detected_text)

    # Step 4: Insert the detected text into the database
    if detected_text:
        insert_license_plate_to_db(detected_text)

if __name__ == "__main__":
    main()
