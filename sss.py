from flask import Flask, request, jsonify
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import re
import os
from datetime import datetime

app = Flask(__name__)

def preprocess_image(image):
    """Enhance image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return denoised

def extract_table_data(image):
    """Extract table data from image using OCR"""
    # Preprocess image
    processed = preprocess_image(image)
    
    # Use tesseract to extract data with table structure
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config=custom_config)
    
    # Organize data by rows
    rows = {}
    for i, text in enumerate(data['text']):
        if text.strip():
            top = data['top'][i]
            # Group by vertical position (within 10 pixels)
            row_key = top // 10
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append({
                'text': text.strip(),
                'left': data['left'][i],
                'conf': data['conf'][i]
            })
    
    # Sort rows and create table
    table_data = []
    for row_key in sorted(rows.keys()):
        # Sort cells in row by left position
        row_cells = sorted(rows[row_key], key=lambda x: x['left'])
        row_text = [cell['text'] for cell in row_cells]
        if row_text:
            table_data.append(row_text)
    
    return table_data

def clean_and_standardize(data):
    """Clean extracted data and create proper DataFrame"""
    if not data:
        return pd.DataFrame()
    
    # Find maximum columns
    max_cols = max(len(row) for row in data)
    
    # Pad rows to have equal columns
    padded_data = []
    for row in data:
        padded_row = row + [''] * (max_cols - len(row))
        padded_data.append(padded_row)
    
    # Create DataFrame
    if len(padded_data) > 1:
        df = pd.DataFrame(padded_data[1:], columns=padded_data[0])
    else:
        df = pd.DataFrame(padded_data)
    
    return df

@app.route('/process', methods=['POST'])
def process_image():
    """API endpoint to process uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        filename = file.filename or 'image.png'
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract table data
        table_data = extract_table_data(image)
        
        # Create DataFrame
        df = clean_and_standardize(table_data)
        
        # Generate CSV content
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"extracted_{timestamp}_{filename.rsplit('.', 1)[0]}.csv"
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'rows': len(df),
            'columns': len(df.columns),
            'csv_data': csv_content,
            'preview': df.head(5).to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Run on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
