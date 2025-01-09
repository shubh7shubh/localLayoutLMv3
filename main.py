import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from paddleocr import PaddleOCR
import numpy as np
import io
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Document Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentProcessor:
    def __init__(self):
        # Initialize OCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        # Label mappings
        self.id2label = {0: 'B-KEY', 1: 'B-VALUE', 2: 'O'}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Initialize model
        self.initialize_model()

    def initialize_model(self):
        try:
            model_path = "input/model/epoch_10"  # Make sure this path exists
            logger.info(f"Loading model from {model_path}")

            self.processor = LayoutLMv3Processor.from_pretrained(model_path)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(self.id2label)
            )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.model.to(self.device)
            self.model.eval()

            logger.info("Model initialization completed successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def process_paddle_ocr_output(self, result, width, height):
        words = []
        boxes = []
        confidences = []

        if not result or not result[0]:
            return [], [], []

        for line in result[0]:
            if len(line) != 2:
                continue

            bbox, (text, conf) = line

            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x1, y1 = bbox[0]
            x2, y2 = bbox[2]

            x1_norm = int((x1 / width) * 1000)
            y1_norm = int((y1 / height) * 1000)
            x2_norm = int((x2 / width) * 1000)
            y2_norm = int((y2 / height) * 1000)

            x1_norm = max(0, min(x1_norm, 1000))
            y1_norm = max(0, min(y1_norm, 1000))
            x2_norm = max(0, min(x2_norm, 1000))
            y2_norm = max(0, min(y2_norm, 1000))

            normalized_box = [x1_norm, y1_norm, x2_norm, y2_norm]

            words.append(text)
            boxes.append(normalized_box)
            confidences.append(conf)

        return words, boxes, confidences

    async def process_pdf(self, pdf_content: bytes):
        """
        Process a PDF file and extract text with layout information
        """
        try:
            # Convert PDF to images
            logger.info("Converting PDF to images")
            # For Windows, specify poppler_path
            images = convert_from_bytes(pdf_content, poppler_path=r"C:/poppler/poppler-24.08.0/Library/bin")

            all_results = []

            for page_num, image in enumerate(images, 1):
                logger.info(f"Processing page {page_num}")

                # Convert PIL Image to bytes for PaddleOCR
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                # Save temporarily for OCR
                temp_image_path = f"temp_page_{page_num}.jpg"
                with open(temp_image_path, "wb") as f:
                    f.write(img_byte_arr)

                width, height = image.size

                # Perform OCR
                ocr_result = self.ocr.ocr(temp_image_path, cls=True)
                words, boxes, confidences = self.process_paddle_ocr_output(ocr_result, width, height)

                if not words:
                    logger.warning(f"No text detected on page {page_num}")
                    continue

                # Process with LayoutLMv3
                try:
                    encoding = self.processor(
                        image,
                        text=words,
                        boxes=boxes,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )

                    encoding = {k: v.to(self.device) for k, v in encoding.items()}

                    with torch.no_grad():
                        outputs = self.model(**encoding)
                        predictions = outputs.logits.argmax(-1)

                    page_results = []
                    for word, box, conf, pred in zip(words, boxes, confidences, predictions[0]):
                        if pred < len(self.id2label):
                            result = {
                                'text': word,
                                'bbox': box,
                                'confidence': float(conf),
                                'label': self.id2label[pred.item()]
                            }
                            page_results.append(result)

                    all_results.append({
                        'page': page_num,
                        'results': page_results
                    })

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {str(e)}")
                    raise

                # Clean up temporary image
                os.remove(temp_image_path)

            return all_results

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

# Initialize processor
processor = DocumentProcessor()

@app.post("/process-pdf/")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    try:
        # Read file content
        pdf_content = await file.read()
        
        # Process the PDF
        results = await processor.process_pdf(pdf_content)
        
        # Prepare response
        response = {
            "filename": file.filename,
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": len(results),
            "total_elements": sum(len(page['results']) for page in results),
            "pages": results
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)