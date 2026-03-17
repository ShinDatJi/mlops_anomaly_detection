import pytest
import requests
import os

api_port = os.environ["MY_API_PORT"]
test_token = os.environ["MY_API_TEST_KEY"]
test_image_file = "tests/bottle_train_good_000.png"

# fixtures

@pytest.fixture(scope="module")
def base_url():
    return f"http://localhost:{api_port}"

@pytest.fixture(scope="module")
def auth_token():
    return test_token

@pytest.fixture(scope="module")
def headers(auth_token):
    return {
        "x-api-key": test_token
    }

@pytest.fixture(scope="module")
def files_func():
    def _files():
        return {'image': open(test_image_file, 'rb')}
    return _files

@pytest.fixture(scope="module")
def predict_endpoint():
    return f"predict/bottle/pretrained"

# Status Test

def test_status(base_url):
    """Test status endpoint"""
    response = requests.get(f"{base_url}/status")
    assert response.status_code == 200
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["status"] == "ok"

# Prediction Test

def test_predict_success(base_url, predict_endpoint, headers, files_func):
    """Test single prediction"""
    files = files_func()
    response = requests.post(f"{base_url}/{predict_endpoint}", files=files, headers=headers)

    assert response.status_code == 200
    prediction = response.json()
    assert isinstance(prediction, dict)

    assert "defective" in prediction
    assert isinstance(prediction["defective"], bool)

    assert "params" in prediction
    assert isinstance(prediction["params"], dict)
    params = prediction["params"]
    assert "patches" in params
    assert "overlap" in params
    assert "height_cropping" in params
    assert "width_cropping" in params
    assert "threshold" in params
    assert isinstance(params["patches"], int)
    assert isinstance(params["overlap"], float)
    assert isinstance(params["height_cropping"], int)
    assert isinstance(params["width_cropping"], int)
    assert isinstance(params["threshold"], float)

    assert "pred_probas" in prediction
    assert isinstance(prediction["pred_probas"], list)
    assert len(prediction["pred_probas"]) > 0
    assert all(isinstance(p, float) for p in prediction["pred_probas"])

def test_predict_unauthorized(base_url, predict_endpoint, files_func):
    """Test prediction with invalid API key"""
    headers = {
        "x-api-key": "invalid_key"
    }
    files = files_func()
    response = requests.post(
        f"{base_url}/{predict_endpoint}",
        files=files,
        headers=headers
    )
    assert response.status_code == 401
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["detail"] == "Unauthorized"

def test_predict_missing_category(base_url, headers, files_func):
    """Test prediction endpoint with missing category in URL"""
    files = files_func()
    response = requests.post(
        f"{base_url}/predict//pretrained",
        files=files,
        headers=headers
    )
    assert response.status_code == 404
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["detail"] == "Not Found"

def test_predict_missing_version(base_url, headers, files_func):
    """Test prediction endpoint with missing version in URL"""
    files = files_func()
    response = requests.post(
        f"{base_url}/predict/bottle/",
        files=files,
        headers=headers
    )
    assert response.status_code == 404
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["detail"] == "Not Found"

def test_predict_wrong_category(base_url, headers, files_func):
    """Test prediction endpoint with an unknown category"""
    files = files_func()
    response = requests.post(
        f"{base_url}/predict/unknown/pretrained",
        files=files,
        headers=headers
    )
    assert response.status_code == 400
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["detail"] == "Unable to load model"

def test_predict_missing_image_file(base_url, predict_endpoint, headers, files_func):
    """Test prediction endpoint with missing image file"""
    files = files_func()
    response = requests.post(
        f"{base_url}/{predict_endpoint}",
        headers=headers
    )
    assert response.status_code == 422
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["detail"][0]["type"] == "missing"
    assert response_data["detail"][0]["msg"] == "Field required"

def test_predict_incorrect_image_file(base_url, predict_endpoint, headers):
    """Test prediction endpoint with incorrect image file"""
    files = {'image': ('test.txt', b'This is not an image file')}
    response = requests.post(
        f"{base_url}/{predict_endpoint}",
        files=files,
        headers=headers
    )
    assert response.status_code == 400
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["detail"] == "Unable to decode image"
