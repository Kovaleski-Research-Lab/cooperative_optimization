import requests
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BASE_URL = "http://128.206.20.44:8000"

def main():

    # 5. Clean up
    r = requests.post(f"{BASE_URL}/reset_bench")
    r.raise_for_status()
    print("Reset bench response:", r.json())
    
    camera_data = {
        "camera_name": "camera0",
        "camera_exposure_time": 100000
    }
    r = requests.post(f"{BASE_URL}/add_camera", json=camera_data)
    r.raise_for_status()
    print("Add camera response:", r.json())

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
    image = io.BytesIO(img_bytes_decoded)
    image = np.load(image)
    
    # Display the image
    fig, ax = plt.subplots(1,1)
    im0 = ax.imshow(image)
    ax.axis("off")
    ax.set_title("Camera Image - Frame 0")

    image_shape = image.shape
    counter = 0
    while(1):
        # 1. Get an image from the camera
        r = requests.get(f"{BASE_URL}/get_camera_image", params={"camera_name": "camera0"})
        r.raise_for_status()
        camera_response = r.json()
        print("Get camera image response received.")
        if "image_data" not in camera_response:
            print("No 'image_data' field found in response. Please ensure the server encodes the image in Base64.")
            return
        img_data = camera_response["image_data"]
        img_bytes_decoded = base64.b64decode(img_data)
        image = io.BytesIO(img_bytes_decoded)
        image = np.load(image)
        print(np.max(image))
        # Display the image
        im0.set_data(image)
        # Draw verticle and horizontal lines in the middle of the image
        ax.axvline(x=image_shape[1]//2, color='red', linestyle='--')
        ax.axhline(y=image_shape[0]//2, color='red', linestyle='--')
        plt.draw()
        plt.pause(0.1)
        counter += 1
        ax.set_title(f"Camera Image - Frame {counter}")
        if counter > 1000:
            break
    
    # 5. Clean up
    r = requests.post(f"{BASE_URL}/reset_bench")
    r.raise_for_status()
    print("Reset bench response:", r.json())

if __name__ == "__main__":
    main()

