import cv2
import numpy as np
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

class EdgeDetector:
    def __init__(self, low_threshold=30, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def detect_edges(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        return edges
    
    def overlay_edges(self, image: np.ndarray, edges):
        color_edges = np.zeros_like(image)
        color_edges[edges > 0] = [0, 0, 255]
        
        result = cv2.addWeighted(image, 0.6, color_edges, 0.4, 0)
        return result


app = FastAPI(docs_url="/docs")

detector = EdgeDetector()

@app.post("/detect-edges")
async def detect_edges(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Только изображения")
    
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except:
        raise HTTPException(400, "Невалидное изображение")
    
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    edges = detector.detect_edges(image_bgr)
    result_image = detector.overlay_edges(image_bgr, edges)
    
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_image_rgb)
    
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)