
import requests
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    url = "http://128.206.20.44:10001"
    info_api_endpoint = "/info"
    response = requests.get(url + info_api_endpoint)
    print(response.text)

    # Add a camera
    camera_params = {
                    'camera_name': 'thorlabs_cc215mu',
                    'camera_exposure_time': 15000,}
    add_camera_api_endpoint = "/add_camera"
    response = requests.post(url + add_camera_api_endpoint, json=camera_params)
    print(response.text)

    # Get an image from the camera
    payload = {'camera_name': 'thorlabs_cc215mu'}
    headers = {'Content-Type': 'application/json'}
    get_image_api_endpoint = "/get_camera_image"
    response = requests.post(url + get_image_api_endpoint, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'ok':
            image = base64.b64decode(data['image'])
            image = Image.open(BytesIO(image))
            image = np.asarray(image)
        else:
            print(f"Error : {data['message']}")
    else:
        print(f"Error : {response.status_code} - {response.reason}")

    # Add an SLM
    slm_params = {
                    'slm_name': 'slm0',
                    'slm_host': '10.10.80.1',}
    add_slm_api_endpoint = "/add_slm"
    response = requests.post(url + add_slm_api_endpoint, json=slm_params)
    print(response.text)

    # Get the bench info
    response = requests.get(url + info_api_endpoint)
    print(response.text)

    # Add an SLM
    slm_params = {
                    'slm_name': 'slm1',
                    'slm_host': '10.10.81.1',}
    add_slm_api_endpoint = "/add_slm"
    response = requests.post(url + add_slm_api_endpoint, json=slm_params)
    print(response.text)

    # Get the bench info
    response = requests.get(url + info_api_endpoint)
    print(response.text)
    # Remove the camera
    camera_params = {
                    'camera_name': 'thorlabs_cc215mu',}
    remove_camera_api_endpoint = "/remove_camera"
    response = requests.post(url + remove_camera_api_endpoint, json=camera_params)
    print(response.text)

    # Get the bench info
    response = requests.get(url + info_api_endpoint)
    print(response.text)


    # Update the slm with an image
    image = np.random.rand(500,500)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    files = {'image': ('image.png', buffered, 'image/png')}
    payload = {'slm_name': 'slm0'}
    update_slm_api_endpoint = "/update_slm"
    response = requests.post(url + update_slm_api_endpoint, files=files, data=payload)
    print(response.text)


    # Remove the SLM
    slm_params = {
                    'slm_name': 'slm0',}
    remove_slm_api_endpoint = "/remove_slm"
    response = requests.post(url + remove_slm_api_endpoint, json=slm_params)
    print(response.text)

    # Get the bench info
    response = requests.get(url + info_api_endpoint)
    print(response.text)

    # Remove the SLM
    slm_params = {
                    'slm_name': 'slm1',}
    remove_slm_api_endpoint = "/remove_slm"
    response = requests.post(url + remove_slm_api_endpoint, json=slm_params)
    print(response.text)

    # Get the bench info
    response = requests.get(url + info_api_endpoint)
    print(response.text)
