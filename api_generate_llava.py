import requests
import json
import base64

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}


def generate_response(prompt,images_path):
    with open(images_path, "rb") as image_file:
    # Convert images to Base64 encoding
        encoded_string = base64.b64encode(image_file.read()).decode()

    data = {
        "model": "llava",
        "stream": False,
        "prompt": prompt,
        "images":[encoded_string]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        print(actual_response)
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return None
    
    
if __name__ == "__main__":
    test_prompt = "who the he is and what he is doing"
    test_img_path = "/mnt/d/Mocheg-main/Mocheg-main/data/test/images/00017-proof-06-GettyImages-1137888397.jpg"
    generate_response(test_prompt,test_img_path)
