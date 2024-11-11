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


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
WORKING_DIR = "./lightrag_index"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

lightrag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  
)

def preprocess_image(image):
    image = image.convert("L") 
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  
    image = image.filter(ImageFilter.SHARPEN) 
    return image

def extract_text_from_pdf(pdf_path):
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            preprocessed_image = preprocess_image(image)
            text += pytesseract.image_to_string(preprocessed_image)
        logger.info("Text extraction successful.")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_pdf_text_based(pdf_path):
    try:
        logger.info(f"Extracting text from text-based PDF: {pdf_path}")
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        logger.info("Text extraction successful.")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from text-based PDF: {str(e)}")
        return ""

def parse_items_from_text(extracted_text):
    items = []
    lines = extracted_text.split("\n")
    

    pattern = re.compile(r"([a-zA-Z\s]+)\s*(\d+)\s*(\d+\.\d{2})|([a-zA-Z\s]+)\s*(\d+\.\d{2})")
    
    for line in lines:
        match = pattern.search(line)
        if match:
            item_name = match.group(1).strip() if match.group(1) else match.group(4).strip()
            quantity = match.group(2) if match.group(2) else "1"  
            price = match.group(3) if match.group(3) else match.group(5)
            items.append({"item_name": item_name, "quantity": quantity, "price": price})
    return items

def classify_items_with_openai(extracted_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4", 
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant that classifies items into medical and non-medical categories."},
                {"role": "user", "content": f"Classify the following items as medical or non-medical:\n{extracted_text}"}
            ]
        )
        classified_items_text = response['choices'][0]['message']['content']
        

        classified_items = []
        for line in classified_items_text.split("\n"):
            parts = line.split("-")
            if len(parts) == 2:
                item_name = parts[0].strip()
                category = parts[1].strip()
                classified_items.append({"item_name": item_name, "category": category})

        logger.info("Classification successful.")
        return classified_items
    except Exception as e:
        logger.error(f"Error classifying items: {str(e)}")
        return []

def store_items_in_lightrag(items):
    for item in items:
        lightrag.insert({"text": item["item_name"], "metadata": {"quantity": item["quantity"], "price": item["price"]}})
    logger.info("Items stored in LightRAG index.")

def retrieve_related_information(query):
    response = lightrag.query(query, param=QueryParam(mode="global"))
    return response

def process_pdf_and_classify(pdf_path):

    if pdf_path.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(pdf_path) if is_scanned_pdf(pdf_path) else extract_text_from_pdf_text_based(pdf_path)
    else:
        logger.error("Unsupported file type.")
        return json.dumps({"status": "error", "message": "Unsupported file type"}, indent=4)
    
    if not extracted_text:
        logger.error("No text extracted from the PDF.")
        return json.dumps({"status": "error", "message": "No text extracted from the PDF"}, indent=4)

    items = parse_items_from_text(extracted_text)

    store_items_in_lightrag(items)

    classified_items = classify_items_with_openai(extracted_text)

    result = {
        "status": "success",
        "extracted_text": extracted_text,
        "classified_items": classified_items,
        "items": items 
    }

    related_info = retrieve_related_information("medical items")
    result["related_info"] = related_info
    
    return json.dumps(result, indent=4)

def is_scanned_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        if images:
            return True
        return False
    except Exception:
        return False
    
def write_items_to_csv(items, filename="extracted_items.csv"):
    try:
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["item_name", "quantity", "price"])
            writer.writeheader()
            writer.writerows(items)
        logger.info(f"Extracted items successfully written to {filename}")
    except Exception as e:
        logger.error(f"Error writing items to CSV: {str(e)}")

def process_pdf_and_classify(pdf_path):
    if pdf_path.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(pdf_path) if is_scanned_pdf(pdf_path) else extract_text_from_pdf_text_based(pdf_path)
    else:
        logger.error("Unsupported file type.")
        return json.dumps({"status": "error", "message": "Unsupported file type"}, indent=4)
    
    if not extracted_text:
        logger.error("No text extracted from the PDF.")
        return json.dumps({"status": "error", "message": "No text extracted from the PDF"}, indent=4)

    items = parse_items_from_text(extracted_text)

    write_items_to_csv(items)
    store_items_in_lightrag(items)

    classified_items = classify_items_with_openai(extracted_text)

    result = {
        "status": "success",
        "extracted_text": extracted_text,
        "classified_items": classified_items,
        "items": items 
    }

    related_info = retrieve_related_information("medical items")
    result["related_info"] = related_info
    
    return json.dumps(result, indent=4)


if __name__ == "__main__":
    pdf_path = r"C:\Users\amank\OneDrive\Desktop\Appp\LightRAG\bill_of_items.pdf" 
    result_json = process_pdf_and_classify(pdf_path)
    print(result_json)
