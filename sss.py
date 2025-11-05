from flask import Flask, request, jsonify
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import os
from datetime import datetime

app = Flask(__name__)

def enhance_image(image):
    """Advanced image enhancement for better OCR"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Increase image size for better OCR (2x)
    scale_factor = 2
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Apply bilateral filter to reduce noise while keeping edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def detect_table_structure(image):
    """Detect table lines and structure"""
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine lines
    table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    
    return table_structure, horizontal_lines, vertical_lines

def extract_cells_by_position(image):
    """Extract table data using TSV output for better structure"""
    # Enhanced tesseract config for table data
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    
    # Get data with bounding boxes
    data = pytesseract.image_to_data(
        image, 
        output_type=pytesseract.Output.DICT, 
        config=custom_config
    )
    
    # Filter out empty text and low confidence
    valid_data = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = float(data['conf'][i])
        
        if text and conf > 30:  # Only accept confidence > 30%
            valid_data.append({
                'text': text,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'conf': conf
            })
    
    if not valid_data:
        return []
    
    # Sort by vertical position first (top to bottom)
    valid_data.sort(key=lambda x: x['top'])
    
    # Group into rows based on vertical position
    rows = []
    current_row = []
    current_top = valid_data[0]['top']
    row_height_threshold = 20  # pixels tolerance for same row
    
    for item in valid_data:
        # If item is roughly at same vertical position, add to current row
        if abs(item['top'] - current_top) < row_height_threshold:
            current_row.append(item)
        else:
            # Sort current row by horizontal position (left to right)
            if current_row:
                current_row.sort(key=lambda x: x['left'])
                rows.append(current_row)
            # Start new row
            current_row = [item]
            current_top = item['top']
    
    # Don't forget the last row
    if current_row:
        current_row.sort(key=lambda x: x['left'])
        rows.append(current_row)
    
    return rows

def align_columns(rows):
    """Align data into proper columns based on horizontal positions"""
    if not rows or len(rows) < 2:
        return rows
    
    # Find all unique left positions (column boundaries)
    all_lefts = set()
    for row in rows:
        for cell in row:
            all_lefts.add(cell['left'])
    
    # Sort column positions
    sorted_positions = sorted(all_lefts)
    
    # Group positions that are close together (within 30 pixels)
    column_positions = []
    current_group = [sorted_positions[0]]
    
    for pos in sorted_positions[1:]:
        if pos - current_group[-1] < 30:
            current_group.append(pos)
        else:
            # Take average of group as column position
            column_positions.append(sum(current_group) // len(current_group))
            current_group = [pos]
    
    if current_group:
        column_positions.append(sum(current_group) // len(current_group))
    
    # Assign each cell to nearest column
    aligned_rows = []
    for row in rows:
        aligned_row = ['' for _ in range(len(column_positions))]
        
        for cell in row:
            # Find nearest column
            distances = [abs(cell['left'] - col_pos) for col_pos in column_positions]
            col_idx = distances.index(min(distances))
            
            # Append to cell (in case multiple texts in same cell)
            if aligned_row[col_idx]:
                aligned_row[col_idx] += ' ' + cell['text']
            else:
                aligned_row[col_idx] = cell['text']
        
        aligned_rows.append(aligned_row)
    
    return aligned_rows

def extract_table_data(image):
    """Main extraction function"""
    # Enhance image
    enhanced = enhance_image(image)
    
    # Extract cells
    rows = extract_cells_by_position(enhanced)
    
    # Align into columns
    table_data = align_columns(rows)
    
    return table_data

def create_dataframe(table_data):
    """Create DataFrame from extracted data"""
    if not table_data or len(table_data) < 2:
        return pd.DataFrame()
    
    # First row as headers
    headers = table_data[0]
    data_rows = table_data[1:]
    
    # Ensure all rows have same number of columns as headers
    max_cols = len(headers)
    cleaned_data = []
    
    for row in data_rows:
        # Pad or trim row to match header length
        if len(row) < max_cols:
            row = row + [''] * (max_cols - len(row))
        elif len(row) > max_cols:
            row = row[:max_cols]
        cleaned_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(cleaned_data, columns=headers)
    
    # Clean up empty rows
    df = df.replace('', np.nan)
    df = df.dropna(how='all')
    
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
        df = create_dataframe(table_data)
        
        if df.empty:
            return jsonify({'error': 'No table data found in image'}), 400
        
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
