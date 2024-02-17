import requests
import json

url = "http://localhost:11434/api/chat"

headers = {
    'Content-Type': 'application/json',
}


def generate_response(prompt):
    
    data = {
        "model": "llama_fc",
        "messages": [
            # {
            #   "role": "system",
            #   "content": "You are a NLP expert that is good at fact checking. You will be presented with a claim and evidence. Your task is to determine whether the claim is supported, refuted, or NEI (for not enough information or partially supported) based on the provided evidence.Your output only needs to provide one of supported, refuted, or NEI. You don't need to give any explaination"
            # },
            {
            "role": "system",
            "content": "You are a professional classifier in fact checking, your job is to be given claims with their evidences and classify them as one of the following categories: supported, refuted, or NEI. If you are unsure and don't have enough info, respond with NEI. Respond only with the classification and nothing else."
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
    
    
