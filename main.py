from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import mimetypes

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
UPLOAD_IMAGE_DIR = os.path.join(BASE_DIR, "uploaded_images")
UPLOAD_VIDEO_DIR = os.path.join(BASE_DIR, "uploaded_videos")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True)
os.makedirs(UPLOAD_VIDEO_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Hello Let's explore the backend here!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()

    if file_extension in ["jpg", "jpeg", "png", "gif"]:
        save_dir = UPLOAD_IMAGE_DIR
    elif file_extension in ["mp4", "avi", "mov", "mkv"]:
        save_dir = UPLOAD_VIDEO_DIR
    else:
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

    file_path = os.path.join(save_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "File uploaded successfully", "filename": file.filename, "file_type": file_extension}

@app.get("/return")
def give_file(filename: str):
    try:
        safe_filename = os.path.basename(filename)  # Prevent directory traversal attacks
        image_path = os.path.abspath(os.path.join(UPLOAD_IMAGE_DIR, safe_filename))
        video_path = os.path.abspath(os.path.join(UPLOAD_VIDEO_DIR, safe_filename))

        if os.path.exists(image_path):
            mime_type, _ = mimetypes.guess_type(image_path)
            return FileResponse(image_path, media_type=mime_type or "application/octet-stream", filename=safe_filename)

        elif os.path.exists(video_path):
            mime_type, _ = mimetypes.guess_type(video_path)
            return FileResponse(video_path, media_type=mime_type or "application/octet-stream", filename=safe_filename)

        return JSONResponse(content={"error": "File not found"}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": "Internal Server Error", "details": str(e)}, status_code=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
