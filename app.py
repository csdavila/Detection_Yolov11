from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import os
import cv2
from ultralytics import YOLO
import datetime
import subprocess

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/runs", StaticFiles(directory="runs"), name="runs")

model = YOLO("yolo11s.pt")

def convert_to_mp4(input_path: str, output_path: str):
    command = [
        "ffmpeg",
        "-y",  
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        output_path
    ]
    subprocess.run(command, check=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result_media": None, "media_type": None})

@app.post("/detect/", response_class=HTMLResponse)
async def detect_media(request: Request, file: UploadFile = File(...), media_type: str = Form(...)):
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_media_path = None

    if media_type == "image":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"detect_{timestamp}"
        results = model(temp_path, save=True, project="runs", name=folder_name, exist_ok=True)
        result_dir = Path(results[0].save_dir)
        image_files = list(result_dir.glob("*.jpg")) + list(result_dir.glob("*.png")) + list(result_dir.glob("*.jpeg"))

        if image_files:
            output_file = image_files[0]
            result_media_path = f"/runs/{output_file.relative_to('runs').as_posix()}"
            print("Ruta imagen generada:", result_media_path)
            
 
    elif media_type == "video":
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_dir = Path("runs/detect")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Guardamos primero en .avi
        out_path_avi = out_dir / "processed_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec avi compatible
        out = cv2.VideoWriter(str(out_path_avi), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        out.release()
        cap.release()

        # Convertimos avi a mp4 compatible con navegador
        out_path_mp4 = out_dir / "processed_video.mp4"
        convert_to_mp4(str(out_path_avi), str(out_path_mp4))

        if out_path_avi.exists():
            out_path_avi.unlink()

        result_media_path = "/runs/detect/processed_video.mp4"
        print("Ruta video generada:", result_media_path)
        

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_media": result_media_path,
        "media_type": media_type
    })
