import requests
import json

url = "http://localhost:11434/api/chat"

headers = {
    "Content-Type": "application/json",
}


def generate_response(prompt):

    data = {
        "model": "llama_fc",
        "messages": [
            # {
            #     "role": "system",
            #     "content": """
            #     You are a professional classifier in fact checking. Your job is to analyze claims with their evidences meticulously. For each claim:
            #     - Classify as 'supported' if evidence corroborates the claim.
            #     - Classify as 'refuted' once its evidence appear to disprove the claim.
            #     - Classify as 'NEI' (Not Enough Information) only if the evidence does not very conclusively support or refute the claim, or if the information presented is quite ambiguous, lacking, or requires more context to make a definitive classification.
            #     Respond only with the classification and nothing else.
            #     """,
            # },
            {
                "role": "system",
                "content": "You are a professional classifier in fact checking, your job is to be given claims with their evidences and classify them as one of the following categories: supported, refuted, or NEI. If you are unsure and don't have enough info, respond with NEI. Respond only with the classification and nothing else."
                # "You are a professional classifier in fact checking, your job is to be given claims with their evidences and classify them as one of the following categories: supported, refuted. Respond only with the classification and nothing else.",
            },
            {
                "role": "user",
                "content": """claim: Using mobile phones leads to the death of bees, threatening the global bee population.
            evidence: A plethora of scientific studies have indicated that the decline in bee populations is related to several factors, including pesticide use, loss of habitat, climate change, etc., but there is no reliable evidence that mobile phone usage directly causes bee deaths.'""",
            },
            {"role": "assistant", "content": "refuted"},
            {
                "role": "user",
                "content": """The ancient Egyptians' technology for building pyramids came from aliens.”
            evidence: While various theories attempt to explain how the ancient Egyptians built the pyramids, including the use of sophisticated engineering techniques and mathematical calculations, there is currently no direct evidence to prove that these technologies were obtained from aliens.""",
            },
            {"role": "assistant", "content": "NEI"},
            {
                "role": "user",
                "content": """claim: Cats can sense earthquakes before they happen.
            evidence: Studies have shown that cats and some other animals exhibit unusual behavior before an earthquake occurs. Scientists believe this might be because they can detect minor pre-earthquake tremors or sounds through their hearing or other senses.""",
            },
            {"role": "assistant", "content": "supported"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        # actual_response = data  #for chat api
        actual_response = data["message"]["content"]  # for chat api
        # print(actual_response)
        return actual_response
    else:
        # print("Error:", response.status_code, response.text)
        return None
