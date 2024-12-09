import requests
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BASE_URL = "http://127.0.0.1:8000"

def main():
    # 1. Add SLMs
    slms_data = [
        {"slm_name": "slm0", "slm_host": "10.10.80.1"},
        {"slm_name": "slm1", "slm_host": "10.10.81.1"}
    ]
    
    for slm in slms_data:
        r = requests.post(f"{BASE_URL}/add_slm", json=slm)
        r.raise_for_status()
        print("Add SLM response:", r.json())
    
    # 2. Add a camera
    camera_data = {
        "camera_name": "camera0",
        "camera_exposure_time": 100000
    }
    r = requests.post(f"{BASE_URL}/add_camera", json=camera_data)
    r.raise_for_status()
    print("Add camera response:", r.json())

    # 3. Update the SLMs with a random image
    # Generate a random image 1920x1080 (width=1920, height=1080)
    # Numpy shape is (height, width, channels)
    height, width = 1080, 1920
    arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    # Save the image to memory for uploading
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # We'll use the same image and same options for both SLMs
    # Adjust "options" and "wait" as needed for your setup
    update_data = {
        "options": ["wl=520", "-fx"],
        "wait": 0.1
    }
    
    for slm_name in ["slm0", "slm1"]:
        files = {
            "image": ("random.png", img_bytes, "image/png")
        }
        data = {
            "slm_name": slm_name,
            "options": update_data["options"],
            "wait": str(update_data["wait"])
        }
        
        r = requests.post(f"{BASE_URL}/update_slm", data=data, files=files)
        r.raise_for_status()
        print(f"Update SLM {slm_name} response:", r.json())
        
        # Reset the BytesIO each time since the request consumes it
        img_bytes.seek(0)

    # 4. Get an image from the camera
    r = requests.get(f"{BASE_URL}/get_camera_image", params={"camera_name": "camera0"})
    r.raise_for_status()
    camera_response = r.json()
    print("Get camera image response received.")

    # Decode the Base64 image
    # The server code from previous examples would return something like:
    # {"status": "ok", "image_data": "<base64_string>"}
    # If your server returns raw data differently, adjust accordingly.
    if "image_data" not in camera_response:
        print("No 'image_data' field found in response. Please ensure the server encodes the image in Base64.")
        return

    img_data = camera_response["image_data"]
    img_bytes_decoded = base64.b64decode(img_data)
    
    # Display the image
    display_img = Image.open(io.BytesIO(img_bytes_decoded))
    plt.imshow(display_img)
    plt.axis("off")
    plt.title("Camera Image")
    plt.show()
    
    # 5. Clean up
    r = requests.post(f"{BASE_URL}/reset_bench")
    r.raise_for_status()
    print("Reset bench response:", r.json())

if __name__ == "__main__":
    main()

