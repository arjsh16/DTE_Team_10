
from fastapi import FastAPI
import uvicorn
import os 

app = FastAPI()

@app.get("/")
def home():
    return {"Hello"}

@app.get("/upload")
def take_image():
    return {"Hello upload is initialized successfully"}

@app.get("/return")
def give_image():
    return {"Hello return is initialized successfully"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

# The above code is a simple FastAPI application with three endpoints:
# 1. `/`: Returns a simple "Hello" message.     
# 2. `/upload`: Returns a message indicating that the upload endpoint is initialized successfully.
# 3. `/return`: Returns a message indicating that the return endpoint is initialized successfully.