"""
FastAPI Application for Managing SLMs and Cameras

This module provides REST API endpoints to:
- Add, remove, and update SLMs (Spatial Light Modulators)
- Add, remove, and update Cameras
- Retrieve camera images
- Reset the entire bench (clear all devices)
- Enforce IP whitelisting and log incoming requests

Author: [Your Name]
"""

from fastapi import FastAPI, File, UploadFile, Request, status, Form, HTTPException
from pydantic import BaseModel
import sys
from typing import List
from io import BytesIO
from PIL import Image
import os
import base64
import numpy as np
import time
import logging

# Add local modules from research directories (adjust these paths as needed)
sys.path.append('/home/mblgh6/Documents/research')
sys.path.append('/home/mblgh6/Documents/research/optics_benchtop')
sys.path.append('/home/mblgh6/Documents/research/cooperative_optimization')

from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu

# Define whitelisted IP addresses allowed to access the API
WHITELISTED_IPS = [
    '127.0.0.1',
    '128.206.23.4',
    '128.206.23.9',
    '128.206.23.10',
    '128.206.20.44',
    '128.206.20.59'
]

# Maximum time (in seconds) to wait for camera image acquisition
TIMEOUT = 5

class SLM(BaseModel):
    """
    Data model for an SLM device.
    """
    slm_name: str
    slm_host: str

class Camera(BaseModel):
    """
    Data model for a Camera device.
    """
    camera_name: str
    camera_exposure_time: int

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Dictionaries to store active SLMs and Cameras
slms = {}
cameras = {}

# Initialize FastAPI app
app = FastAPI()

@app.middleware('http')
async def validate_ip(request: Request, call_next):
    """
    Middleware to validate incoming requests against the whitelist.
    """
    ip = str(request.client.host)
    if ip not in WHITELISTED_IPS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"IP {ip} is not allowed to access this resource."
        )
    return await call_next(request)

@app.middleware('http')
async def log_requests(request: Request, call_next):
    """
    Middleware to log incoming requests and their responses.
    """
    logging.info(f"Received request from {request.client.host} for {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

@app.post('/add_slm')
def add_slm(slm: SLM):
    """
    Endpoint to add a new SLM.
    """
    slm_name = slm.slm_name
    slm_host = slm.slm_host

    if slm_name in slms:
        raise HTTPException(status_code=400, detail="SLM already exists")
    
    # Create and store the SLM object
    slm_obj = holoeye_pluto21.HoloeyePluto(host=slm_host)
    slms[slm_name] = slm_obj
    return {"status": "SLM added successfully"}

@app.post('/add_camera')
def add_camera(camera: Camera):
    """
    Endpoint to add a new Camera.
    """
    camera_name = camera.camera_name
    camera_exposure_time = camera.camera_exposure_time

    if camera_name in cameras:
        raise HTTPException(status_code=400, detail="Camera already exists")
    
    # Create and store the Camera object
    camera_obj = thorlabs_cc215mu.Thorlabs_CC215MU(exposure_time_us=int(camera_exposure_time))
    cameras[camera_name] = camera_obj
    return {"status": "Camera added successfully"}

@app.post('/remove_slm')
def remove_slm(slm_name: str):
    """
    Endpoint to remove an existing SLM.
    """
    if slm_name not in slms:
        raise HTTPException(status_code=400, detail="SLM does not exist")
    
    del slms[slm_name]
    return {"status": "SLM removed successfully"}

@app.post('/remove_camera')
def remove_camera(camera_name: str):
    """
    Endpoint to remove an existing Camera.
    """
    if camera_name not in cameras:
        raise HTTPException(status_code=400, detail="Camera does not exist")
    
    cameras[camera_name].clean_up()
    del cameras[camera_name]
    return {"status": "Camera removed successfully"}

@app.post('/update_slm')
async def update_slm(
    slm_name: str = Form(...),
    options: List[str] = Form(...),
    wait: float = Form(...),
    image: UploadFile = File(...)
):
    """
    Endpoint to upload and update an SLM with a new image.
    """
    if not slm_name or not options or wait is None:
        raise HTTPException(status_code=400, detail="Missing parameters")

    if slm_name not in slms:
        raise HTTPException(status_code=400, detail=f"SLM '{slm_name}' not found")

    try:
        # Load the uploaded image
        img = Image.open(BytesIO(await image.read()))
        temp_filename = f"{slm_name}_temp.png"
        img.save(temp_filename)

        # Transfer and update the SLM
        slms[slm_name].send_scp(temp_filename)
        slms[slm_name].update(filename=f"/root/{temp_filename}", options=options, wait=wait)

        # Remove the temporary file
        os.remove(temp_filename)
        return {"status": "ok", "message": "Image received successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post('/update_camera')
def update_camera(camera_name: str, camera_exposure_time: int):
    """
    Endpoint to update an existing camera's exposure time.
    """
    if camera_name not in cameras:
        raise HTTPException(status_code=400, detail="Camera does not exist")

    cameras[camera_name].update_exposure_time(camera_exposure_time)
    return {"status": "ok", "message": "Camera exposure time updated successfully"}

@app.get('/get_camera_image')
def get_camera_image(camera_name: str):
    """
    Endpoint to capture and retrieve a camera image (Base64-encoded).
    """
    if camera_name not in cameras:
        raise HTTPException(status_code=400, detail="Camera does not exist")

    try:
        image = None
        # Issue a software trigger to the camera
        cameras[camera_name].camera.issue_software_trigger()

        start_time = time.time()
        while image is None:
            image = cameras[camera_name].get_image(pil_image=True, eight_bit=True)
            if time.time() - start_time > TIMEOUT:
                raise HTTPException(status_code=500, detail="Timeout occurred while waiting for camera image")

        # Serialize the image using NumPy and Base64
        buffer = BytesIO()
        np.save(buffer, image)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {"status": "ok", "image_data": img_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post('/reset_bench')
def reset_bench():
    """
    Endpoint to reset all devices (SLMs and Cameras).
    """
    slms.clear()
    for camera in cameras.values():
        camera.clean_up()
    cameras.clear()
    return {"status": "ok", "message": "Bench reset successfully"}

@app.get('/info')
def info():
    """
    Endpoint to get information about connected SLMs and Cameras.
    """
    return {"slms": list(slms.keys()), "cameras": list(cameras.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="128.206.20.44", port=8000)

