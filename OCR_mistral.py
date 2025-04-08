import base64
import requests
import os
from mistralai import Mistral

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

def append_to_markdown(file_path, content):
    """Append content to the end of a markdown file."""
    try:
        with open(file_path, "a") as md_file:
            md_file.write(content + "\n")
    except Exception as e:
        print(f"Error: {e}")

# Path to your image
image_path = "methods/1.png"

# Getting the base64 string
base64_image = encode_image(image_path)
if base64_image is None:
    exit(1)  # Exit if encoding failed

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{base64_image}"
    }
)
print(ocr_response)
# Assuming ocr_response contains the result in a readable format
# result_text = ocr_response.get("text", "No text found")

# # Path to your markdown file
# markdown_path = "results.md"

# # Append the result to the markdown file
# append_to_markdown(markdown_path, result_text)

# print("OCR result appended to markdown file.")
