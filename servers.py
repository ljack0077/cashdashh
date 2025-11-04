import os
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# ============================================
# GPU OPTIMIZATION CONFIGURATION
# ============================================
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Enable GPU optimizations
if torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(0.90, device=0)
    except Exception as e:
        print(f"⚠️ Could not set memory fraction: {e}")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"✅ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✅ Allocated for use: ~{torch.cuda.get_device_properties(0).total_memory * 0.90 / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: No GPU detected! Running on CPU.")

# ============================================
# CONFIGURE DOCLING FOR MAXIMUM ACCURACY
# ============================================
pipeline_options = PdfPipelineOptions()

# Enable table structure recognition with maximum accuracy
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.table_structure_options.do_cell_matching = True

# Increase image quality for better OCR/table detection
pipeline_options.images_scale = 2.0
pipeline_options.generate_page_images = True

# OCR settings
pipeline_options.do_ocr = True

# Create the DocumentConverter with optimized settings
# FIXED: Removed the backend parameter - let Docling choose automatically
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
            # REMOVED: backend="pypdfium2" - causes validation error
        )
    }
)

print("✅ Docling converter initialized with high-accuracy settings")

# ============================================
# FASTAPI APPLICATION
# ============================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "total_vram_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}",
            "allocated_vram_gb": f"{torch.cuda.memory_allocated() / 1e9:.2f}",
            "reserved_vram_gb": f"{torch.cuda.memory_reserved() / 1e9:.2f}",
        }
    
    return {
        "service": "Optimized Docling CSV Server",
        "author": "11001",
        "endpoint": "POST /extract - Upload image, get CSV",
        "gpu_enabled": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "accuracy_mode": "HIGH (TableFormerMode.ACCURATE)",
        "table_cell_matching": "ENABLED",
        "image_scale": "2.0x"
    }

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Upload image → Get CSV back with optimized GPU processing"""
    
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        if torch.cuda.is_available():
            print(f"🔍 GPU Memory before processing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        result = converter.convert(tmp_path)
        
        if torch.cuda.is_available():
            print(f"🔍 GPU Memory after processing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"🔍 GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # Get all tables
        all_data = []
        for table in result.document.tables:
            if hasattr(table, 'data') and table.data:
                df = table.export_to_dataframe()
                all_data.append(df)
        
        if not all_data:
            return Response(
                content="No tables found", 
                media_type="text/plain", 
                status_code=400
            )
        
        combined = pd.concat(all_data, ignore_index=True)
        combined.insert(0, 'timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        csv_data = combined.to_csv(index=False, encoding='utf-8-sig')
        
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=extracted.csv"}
        )
    
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response(
            content=f"Error processing file: {str(e)}", 
            media_type="text/plain", 
            status_code=500
        )
    
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import uvicorn
    
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("🚀 SERVER STARTING WITH GPU ACCELERATION")
        print("="*60)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available for processing: ~{torch.cuda.get_device_properties(0).total_memory * 0.90 / 1e9:.2f} GB")
        print(f"Table Recognition Mode: ACCURATE (maximum quality)")
        print(f"Image Scale: 2.0x (high resolution)")
        print(f"Cell Matching: ENABLED")
        print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8003)
