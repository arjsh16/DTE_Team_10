
from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
import os 

app = FastAPI()
UPLOAD_DIR = "uploaded_image"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"Hello"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully", "filename": file.filename}
@app.get("/return")
def give_image():
    return {"Hello return is initialized successfully"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
# Local address is 127.0.0.1:8080 for home page, 127.0.0.1:8080/upload for taking image, 127.0.0.1:8080/return for giving the processed image.
# update @ 10:20am 29/03/25: Created the basic structure of the FastAPI app. Will now move forward to create the noise reduction and image generation function.
# update @ 02:40pm 29/03/25: is taking images as an input
