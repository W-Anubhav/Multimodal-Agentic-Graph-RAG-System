import torch
import os
from PIL import Image
from pdf2image import convert_from_path
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TableTransformerForObjectDetection
)
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#  Hardware Setup
device = "cuda" if torch.cuda.is_available() else "cpu"


# LayoutLMv3 (General Text & Layout)
layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to(device)

# TATR (Table Structure Specialist)
table_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
).to(device)

def process_file(file_path):
    """Handles both multi-page PDFs and single images."""
    images = []
    if file_path.lower().endswith(".pdf"):
        
        images = convert_from_path(file_path, poppler_path=r"C:\Users\anubhav maurya\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin")
    else:
        images = [Image.open(file_path).convert("RGB")]
    
    document_data = []
    for page_num, img in enumerate(images):
        print(f"Extracting Page {page_num + 1}...")
        page_result = extract_page_data(img, page_num + 1)
        document_data.append(page_result)
        
    return document_data

def extract_page_data(image, page_num):
    """Runs both LayoutLMv3 and TATR on a single page."""
    
    # STAGE 1: LayoutLMv3 (Words & General Bounding Boxes) 
    encoding = layout_processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        layout_outputs = layout_model(**encoding)
    
    # Extract the actual words and their [0-1000] normalized coordinates
    words = layout_processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    word_boxes = encoding["bbox"][0].tolist()
    
    # Clean up WordPiece tokens (removing '##' fragments for cleaner data)
    clean_words = []
    clean_boxes = []
    for w, b in zip(words, word_boxes):
        if w not in ["<s>", "</s>", "<pad>"] and not w.startswith("##"):
            clean_words.append(w)
            clean_boxes.append(b)

    #  STAGE 2: TATR (Table Grid Coordinates) 
    # We use the same pixel values prepared by the processor
    with torch.no_grad():
        table_outputs = table_model(encoding["pixel_values"])
    
    # Format the output into a clean dictionary
    return {
        "page_number": page_num,
        "text_data": {
            "words": clean_words,
            "boxes": clean_boxes
        },
        "table_data": {
            "boxes": table_outputs.pred_boxes.cpu().tolist(),
            "labels": table_outputs.logits.argmax(-1).cpu().tolist()
        }
    }

if __name__ == "__main__":
    
    pass