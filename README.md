# Medical Item Classification from Bill of Items

This project extracts item details (name, quantity, price) from a scanned bill of items, classifies them as medical or non-medical using OpenAI's GPT-4 model, and stores the extracted data in LightRAG's vector database for future retrieval. Additionally, it exports the extracted data to a CSV file.

## Table of Contents
- [Objective](#objective)
- [Tools Used](#tools-used)

## Objective

The goal of this project is to:
1. **Extract item details (name, quantity, price)** from a scanned or text-based bill of items.
2. **Classify the items into medical and non-medical categories** using OpenAI's GPT-4 model.
3. **Store and retrieve extracted items** using LightRAG, a graph-based tool.
4. **Export extracted item data into a CSV file** for easy viewing and processing.

## Tools Used

- **OpenAI GPT-4**: Used for classifying items as medical or non-medical.
- **LightRAG**: A graph-based RAG tool to store and retrieve the extracted item details.
- **Pytesseract**: Optical Character Recognition (OCR) library to extract text from scanned PDF images.
- **pdf2image**: Converts PDFs into images for OCR processing.
- **PyPDF2**: Extracts text from text-based PDFs.
- **Pillow**: Image processing for better OCR accuracy (contrast enhancement, sharpening).
- **dotenv**: Loads environment variables, such as OpenAI API keys.
