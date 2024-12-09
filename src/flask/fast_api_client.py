import httpx

def test_fastapi_server():
    base_url = "http://127.0.0.1:8000"  # Replace with your server's address if different

    # Test the GET endpoint
    print("Testing GET / endpoint...")
    response = httpx.get(f"{base_url}/")
    if response.status_code == 200:
        print("GET / response:", response.json())
    else:
        print("GET / failed with status code:", response.status_code)

    # Test the POST /add_slm endpoint
    print("\nTesting POST /add_slm endpoint...")
    slm_data = {
        "slm_name": "SLM-Test",
        "slm_host": "192.168.1.1"
    }
    response = httpx.post(f"{base_url}/add_slm", json=slm_data)
    if response.status_code == 200:
        print("POST /add_slm response:", response.json())
    else:
        print("POST /add_slm failed with status code:", response.status_code)
        print("Response body:", response.text)

if __name__ == "__main__":
    test_fastapi_server()

