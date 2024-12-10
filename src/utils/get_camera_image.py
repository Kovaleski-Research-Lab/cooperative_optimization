
import requests
import base64
import numpy as np
import io
import matplotlib.pyplot as plt

#
def get_camera_image(camera_name, bench_server_url, get_camera_image_endpoint):
    # Get an image from the camera

    r = requests.get(bench_server_url + get_camera_image_endpoint, params={"camera_name": camera_name})
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
    return image


if __name__ == "__main__":
    bench_server_url = "http://128.206.20.44:8000"
    get_camera_image_endpoint = "/get_camera_image"
    camera_name = "THORLABS_CC215MU"
    image = get_camera_image(camera_name, bench_server_url, get_camera_image_endpoint)
    plt.imshow(image)
    plt.show()
