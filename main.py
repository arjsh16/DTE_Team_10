    
from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import mimetypes
import numpy as np
from image_processing import process_file

app = FastAPI()

# Allow frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
BASE_DIR = "Arithmania"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Helper function to convert NumPy values to Python native types
def convert_numpy_to_python(data):
    """Convert NumPy data types to Python native types for JSON serialization"""
    if isinstance(data, np.number):
        return data.item()  # Convert to Python scalar
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert to Python list
    elif isinstance(data, dict):
        return {k: convert_numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        return [convert_numpy_to_python(item) for item in data]
    else:
        return data

@app.get("/")
def home():
    return {"message": "Hello Let's explore the backend!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()
    
    # Check if file format is supported
    if file_extension not in ["jpg", "jpeg", "png", "gif", "mp4", "avi", "mov", "mkv"]:
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully", 
        "filename": file.filename, 
        "file_type": file_extension,
        "file_path": file_path
    }

@app.post("/process")
async def process_media(
    filename: str = Form(...),
    method: str = Form("adaptive"),
    h: float = Form(10.0),
    color_h: float = Form(10.0),
    d: int = Form(9),
    sigma: float = Form(35.0),
    filter_strength: int = Form(8),
    weight: float = Form(35.0),
    iterations: int = Form(20)
):
    # Create safe filename and find the uploaded file
    safe_filename = os.path.basename(filename)
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(input_path):
        return JSONResponse(content={"error": f"File {safe_filename} not found"}, status_code=404)
    
    # Create output path in processed directory
    base_name, ext = os.path.splitext(safe_filename)
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.tif', '.bmp']:
        # For images, always save as PNG
        output_path = os.path.join(PROCESSED_DIR, f"{base_name}_{method}_processed.png")
    else:
        # For videos, preserve original extension
        output_path = os.path.join(PROCESSED_DIR, f"{base_name}_{method}_processed{ext}")
    
    try:
        # Process the file using our imported function
        result_path, metrics = process_file(
            input_path, 
            output_path,
            method=method,
            h=h,
            color_h=color_h,
            d=d,
            sigma=sigma,
            filter_strength=filter_strength,
            weight=weight,
            iterations=iterations
        )
        
        if result_path is None:
            return JSONResponse(
                content={"error": "Processing failed"}, 
                status_code=500
            )
            
        # Return the processed filename and metrics
        processed_filename = os.path.basename(result_path)
        
        # Convert any NumPy values in metrics to Python native types
        metrics = convert_numpy_to_python(metrics)
        
        return {
            "message": "Media processed successfully",
            "processed_file": processed_filename,
            "metrics": metrics
        }
    
    except Exception as e:
        return JSONResponse(
            content={"error": "Error processing media", "details": str(e)},
            status_code=500
        )

@app.get("/files/{filename}")
def get_file(filename: str):
    try:
        safe_filename = os.path.basename(filename)  # Prevent directory traversal attacks
        
        # First check processed directory
        processed_path = os.path.join(PROCESSED_DIR, safe_filename)
        if os.path.exists(processed_path):
            mime_type, _ = mimetypes.guess_type(processed_path)
            return FileResponse(
                processed_path, 
                media_type=mime_type or "application/octet-stream", 
                filename=safe_filename
            )
            
        # Then check upload directory
        upload_path = os.path.join(UPLOAD_DIR, safe_filename)
        if os.path.exists(upload_path):
            mime_type, _ = mimetypes.guess_type(upload_path)
            return FileResponse(
                upload_path, 
                media_type=mime_type or "application/octet-stream", 
                filename=safe_filename
            )
            
        # If we get here, the file was not found in either directory
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": "Internal Server Error", "details": str(e)}, status_code=500)

@app.get("/list-files")
def list_files(type: str = "all"):
    """List all available files"""
    files = {}
    
    if type in ["all", "uploaded"]:
        files["uploads"] = os.listdir(UPLOAD_DIR)
    
    if type in ["all", "processed"]:
        files["processed"] = os.listdir(PROCESSED_DIR)
    
    return files

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
