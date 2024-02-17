import requests
import json
import base64

url = "http://localhost:11434/api/chat"

headers = {
    'Content-Type': 'application/json',
}


def generate_response(prompt,images_path):
    with open(images_path, "rb") as image_file:
    # Convert images to Base64 encoding
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    data = {
        "model": "llava_fc",
        "messages": [
            {
            "role": "system",
            "content":"You are a professional categorizer in fact checking and your job is to get an image and the claim with its evidence and categorize the claim into one of the following categories: support, refute or NEI.If you are not sure and do not have enough information, reply with 'NEI'. Answer only the categorization and nothing else. " 
            # "You are a professional classifier in fact checking, your job is to be given claims with their evidences and classify them as one of the following categories: supported, refuted, or NEI. If you are unsure and don't have enough info, respond with NEI. Respond only with the classification and nothing else."
            },
            {
            "role": "user",
            "content": """claim: Elephants can fly.
            evidence: There's no scientific evidence or biological basis for elephants being able to fly."""
            },
            {
            "role": "assistant",
            "content": "refuted"
            },
            {
            "role": "user",
            "content": """A specific species of frog can live for 200 years.
            evidence: There is limited research on the lifespan of this frog species."""
            },
            {
            "role": "assistant",
            "content": "NEI"
            },
            {
            "role": "user",
            "content": """claim: The internet was invented in the 1960s.
            evidence: Historical records show the development of ARPANET, a precursor to the internet, began in the late 1960s."""
            },
            {
            "role": "assistant",
            "content": "supported"
            },
            {
            "role": "user",
            "content": prompt
            }
            ],
        
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        # actual_response = data  #for chat api
        actual_response = data["message"]["content"]  #for chat api
        # print(actual_response)
        return actual_response
    else:
        # print("Error:", response.status_code, response.text)
        return None
    
    
if __name__ == "__main__":
    test_prompt = "who is that in the picture"
    test_img_path = "/mnt/d/Mocheg-main/Mocheg-main/data/00017-530390-06-GettyImages-1137888397.jpg"
    generate_response(test_prompt,test_img_path)
    