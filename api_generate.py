import requests
import json

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}


def generate_response(prompt):
    

    data = {
        "model": "fc_llama",
        "stream": False,
        "prompt": prompt,
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
    test_prompt = " claim: The internet was invented in the 1960s.evidence: Historical records show the development of ARPANET, a precursor to the internet, began in the late 1960s."
    generate_response(test_prompt)
