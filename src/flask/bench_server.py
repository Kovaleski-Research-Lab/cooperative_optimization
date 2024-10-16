
from flask import Flask, request, jsonify, abort, send_file
import requests
from bson import ObjectId
import threading
import sys
import base64
from PIL import Image
from io import BytesIO
import numpy as np

sys.path.append('/home/mblgh6/Documents/research')
sys.path.append('/home/mblgh6/Documents/research/optics_benchtop')
sys.path.append('/home/mblgh6/Documents/research/cooperative_optimization')
from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu

trusted_ips = ['128.206.23.4', '128.206.23.9', '128.206.23.10', '128.206.20.44', '128.206.20.59'] 

app = Flask(__name__)


slms = {}
cameras = {}

@app.before_request
def limit_remote_addr():
    if request.remote_addr not in trusted_ips:
        abort(403)

@app.route('/add_slm', methods=['POST'])
def add_slm():
    data = request.get_json()
    keys = data.keys()
    if 'slm_name' not in keys or 'slm_host' not in keys:
        return jsonify({'status': 'error', 'message': 'Missing parameters'})

    slm_name = data['slm_name']
    slm_host = data['slm_host']
    if slm_name in slms:
        return jsonify({'status': 'error', 'message': 'SLM already exists'})
    else:
        slm = holoeye_pluto21.HoloeyePluto(host=slm_host)
        slms[slm_name] = slm
        return jsonify({'status': 'ok'})

@app.route('/add_camera', methods=['POST'])
def add_camera():
    data = request.get_json()
    keys = data.keys()
    if 'camera_name' not in keys or 'camera_exposure_time' not in keys:
        return jsonify({'status': 'error', 'message': 'Missing parameters'})
    camera_name = data['camera_name']
    if camera_name in cameras:
        return jsonify({'status': 'error', 'message': 'Camera already exists'})
    else:
        camera_exposure_time = int(data['camera_exposure_time'])
        camera = thorlabs_cc215mu.Thorlabs_CC215MU(exposure_time_us=camera_exposure_time)
        cameras[camera_name] = camera
        return jsonify({'status': 'ok'})

@app.route('/remove_slm', methods=['POST'])
def remove_slm():
    data = request.get_json()
    keys = data.keys()
    if 'slm_name' not in keys:
        return jsonify({'status': 'error', 'message': 'Missing parameters'})
    slm_name = data['slm_name']
    if slm_name in slms:
        del slms[slm_name]
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'message': 'SLM does not exist'})

@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    data = request.get_json()
    keys = data.keys()
    if 'camera_name' not in keys:
        return jsonify({'status': 'error', 'message': 'Missing parameters'})
    camera_name = data['camera_name']
    if camera_name in cameras:
        cameras[camera_name].clean_up()
        del cameras[camera_name]
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'message': 'Camera does not exist'})

@app.route('/update_slm', methods=['POST'])
def update_slm():
    if 'slm_name' not in request.form:
        return jsonify({'status': 'error', 'message': 'Missing parameters'})

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    file = request.files['image']

    try:
        # Convert the uploaded image (binary data) to a NumPy array
        img = Image.open(BytesIO(file.read()))
        img_np = np.array(img)
        return jsonify({'status': 'ok', 'message': 'Image received successfully', 'shape': img_np.shape})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing image: {str(e)}'}), 500

@app.route('/update_camera', methods=['POST'])
def update_camera():
    data = request.get_json()
    keys = data.keys()
    if 'camera_name' not in keys or 'camera_exposure_time' not in keys:
        return jsonify({'status': 'error', 'message': 'Missing parameters'})
    camera_name = data['camera_name']
    camera_exposure_time = int(data['camera_exposure_time'])
    if camera_name in cameras:
        cameras[camera_name].update_exposure_time(camera_exposure_time)
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'message': 'Camera does not exist'})

@app.route('/get_camera_image', methods=['POST'])  # Use POST to expect JSON body
def get_camera_image():
    try:
        # Check if request content type is JSON
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Invalid content type, must be JSON'}), 400

        data = request.get_json()

        # Ensure 'camera_name' parameter is present
        if 'camera_name' not in data:
            return jsonify({'status': 'error', 'message': 'Missing "camera_name" parameter'}), 400

        camera_name = data['camera_name']

        # Check if the camera exists in the dictionary
        if camera_name not in cameras:
            return jsonify({'status': 'error', 'message': 'Camera does not exist'}), 404

        # Trigger the camera and retrieve the image
        image = None
        cameras[camera_name].camera.issue_software_trigger()
        while image is None:
            image = cameras[camera_name].get_image(pil_image=False, eight_bit=False)

        # Convert NumPy array to PIL image (preserve data format)
        pil_image = Image.fromarray(np.uint8(image))

        # Convert the image to PNG format without compression and encode as base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")  # PNG format is lossless
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Return the base64-encoded image
        return jsonify({'status': 'ok', 'image': img_str})

    except KeyError as e:
        return jsonify({'status': 'error', 'message': f'Missing key: {str(e)}'}), 400

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@app.route('/reset_bench', methods=['POST'])
def reset_bench():
    slms.clear()
    for camera in cameras.values():
        camera.clean_up()
    cameras.clear()
    return jsonify({'status': 'ok'})

@app.route('/info', methods=['GET'])
def info():
    return jsonify({'slms': list(slms.keys()), 'cameras': list(cameras.keys())})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10001)
