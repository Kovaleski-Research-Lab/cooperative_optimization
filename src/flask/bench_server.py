
from flask import Flask, request, jsonify, abort, send_file
import requests
from bson import ObjectId
import threading
import sys

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
    slm_name = data['slm_name']
    if slm_name in slms:
        del slms[slm_name]
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'message': 'SLM does not exist'})

@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    data = request.get_json()
    camera_name = data['camera_name']
    if camera_name in cameras:
        cameras[camera_name].cleanup()
        del cameras[camera_name]
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'message': 'Camera does not exist'})

@app.route('/update_slm', methods=['POST'])
def update_slm():
    data = request.get_json()
    slm_name = data['slm_name']
    slm_host = data['slm_host']
    if slm_name in slms:
        slms[slm_name].host = slm_host
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'message': 'SLM does not exist'})

@app.route('/update_camera', methods=['POST'])
def update_camera():
    data = request.get_json()
    camera_name = data['camera_name']
    camera_exposure_time = int(data['camera_exposure_time'])
    if camera_name in cameras:
        cameras[camera_name].update_exposure_time(camera_exposure_time)
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'message': 'Camera does not exist'})

@app.route('/get_camera_image', methods=['GET'])
def get_camera_image():
    data = request.get_json()
    camera_name = data['camera_name']
    if camera_name in cameras:
        image = None
        cameras[camera_name].camera.issue_software_trigger()
        while image is None:
            image = cameras[camera_name].get_image(pil_image = False, eight_bit = False)
        return jsonify({'status': 'ok', 'image': image.tolist()})
    else:
        return jsonify({'status': 'error', 'message': 'Camera does not exist'})

@app.route('/info', methods=['GET'])
def info():
    return jsonify({'slms': list(slms.keys()), 'cameras': list(cameras.keys())})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10001)
