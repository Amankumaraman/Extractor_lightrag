import os
import json
import openai
import logging
import pytesseract
from pdf2image import convert_from_path
import re
import PyPDF2
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
import csv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LightRAG
WORKING_DIR = "./lightrag_index"
os.makedirs(WORKING_DIR, exist_ok=True)

lightrag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete
)

# Image processing for OCR
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = ImageEnhance.Contrast(image).enhance(2.0)  # Enhance contrast
    return image.filter(ImageFilter.SHARPEN)  # Sharpen image

# Extract text from scanned PDFs using OCR
def extract_text_from_scanned_pdf(pdf_path):
    try:
        logger.info(f"Processing scanned PDF: {pdf_path}")
        images = convert_from_path(pdf_path)
        text = ''.join(pytesseract.image_to_string(preprocess_image(image)) for image in images)
        logger.info("OCR completed.")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from scanned PDF: {e}")
        return ""

# Extract text from text-based PDFs
def extract_text_from_text_pdf(pdf_path):
    try:
        logger.info(f"Processing text-based PDF: {pdf_path}")
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return ''.join(page.extract_text() for page in reader.pages).strip()
    except Exception as e:
        logger.error(f"Error extracting text from text-based PDF: {e}")
        return ""

# Detect whether a PDF is scanned
def is_scanned_pdf(pdf_path):
    try:
        return bool(convert_from_path(pdf_path))
    except Exception:
        return False

# Parse items from extracted text
def parse_items(extracted_text):
    pattern = re.compile(r"([a-zA-Z\s]+)\s*(\d+)?\s*(\d+\.\d{2})")
    items = []
    for line in extracted_text.split("\n"):
        match = pattern.search(line)
        if match:
            items.append({
                "item_name": match.group(1).strip(),
                "quantity": match.group(2) or "1",  # Default to 1 if missing
                "price": match.group(3)
            })
    return items

# Classify items using OpenAI
def classify_items_with_openai(items):
    try:
        prompt = "\n".join(f"{item['item_name']} - Quantity: {item['quantity']}, Price: {item['price']}" for item in items)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Classify the following items as medical or non-medical."},
                {"role": "user", "content": prompt}
            ]
        )
        classifications = response['choices'][0]['message']['content'].strip()
        return [{"item": line.split("-")[0].strip(), "category": line.split("-")[1].strip()} 
                for line in classifications.split("\n") if "-" in line]
    except Exception as e:
        logger.error(f"Error classifying items: {e}")
        return []

# Write extracted items to CSV
def write_items_to_csv(items, filename="extracted_items.csv"):
    try:
        with open(filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["item_name", "quantity", "price"])
            writer.writeheader()
            writer.writerows(items)
        logger.info(f"Items written to {filename}.")
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")

# Main processing function
def process_pdf(pdf_path):
    if not pdf_path.endswith(".pdf"):
        logger.error("Unsupported file type.")
        return {"status": "error", "message": "Unsupported file type"}

    extracted_text = (extract_text_from_scanned_pdf(pdf_path) 
                      if is_scanned_pdf(pdf_path) 
                      else extract_text_from_text_pdf(pdf_path))
    
    if not extracted_text:
        logger.error("No text extracted.")
        return {"status": "error", "message": "No text extracted"}

    items = parse_items(extracted_text)
    write_items_to_csv(items)
    store_items_in_lightrag(items)
    
    classified_items = classify_items_with_openai(items)
    return {
        "status": "success",
        "parsed_items": items,
    }

# Store parsed items in LightRAG
def store_items_in_lightrag(items):
    try:
        for item in items:
            lightrag.insert({"text": item["item_name"], "metadata": {"quantity": item["quantity"], "price": item["price"]}})
        logger.info("Items stored in LightRAG.")
    except Exception as e:
        logger.error(f"Error storing items in LightRAG: {e}")

# Main execution
if __name__ == "__main__":
    pdf_path = r"C:\Users\amank\OneDrive\Desktop\Appp\LightRAG\bill_of_items.pdf"  
    result = process_pdf(pdf_path)
    print(json.dumps(result, indent=4))
