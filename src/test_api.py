import requests
import json

# 1. The URL where your Flask app is running
API_URL = "http://localhost:5000/predict"

# 2. Path to a test image (Change this to any image on your computer)
# Make sure this image actually exists!
IMAGE_PATH = "data/Country_wise_Dataset/India/Images/some_test_image.jpg" 

def test_prediction():
    print(f"🚀 Sending image to API: {API_URL}")
    print(f"📷 Using image: {IMAGE_PATH}\n")

    try:
        # Open the image file in binary read mode ('rb')
        with open(IMAGE_PATH, 'rb') as img_file:
            # The key 'image' must match what we look for in app.py: request.files['image']
            files = {'image': img_file}
            
            # Send the POST request
            response = requests.post(API_URL, files=files)
            
            # Check the response status code (200 means OK)
            if response.status_code == 200:
                print("✅ Success! Here is the model's prediction:")
                # Pretty-print the JSON response
                print(json.dumps(response.json(), indent=4))
            else:
                print(f"❌ API returned an error (Status Code: {response.status_code})")
                print("Error details:", response.text)

    except FileNotFoundError:
        print(f"❌ Error: Could not find the image at '{IMAGE_PATH}'.")
        print("Please update the IMAGE_PATH variable with a valid image path.")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("Did you forget to start the Flask app? (Run `python app.py` first!)")

if __name__ == "__main__":
    test_prediction()