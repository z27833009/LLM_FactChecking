import requests
import json
import base64

url = "http://localhost:11434/api/chat"

headers = {
    "Content-Type": "application/json",
}


def generate_response(prompt, images_path=None):

    messages = [
        {
            "role": "system",
            "content": "You are a professional classifier in fact-checking. Your job is to receive claims along with their evidences, which now include both text and images, and classify them into one of the following categories: supported, refuted, or NEI (Not Enough Information). Respond with NEI only if you are unsure or lack sufficient information. Respond only with the classification and nothing else.",            
            # "You are a professional classifier in fact-checking. Your job is to receive claims along with their evidences, which now include both text and images, and classify them into one of the following categories: supported, refuted, or NEI (Not Enough Information). Respond with NEI only if you are unsure or lack sufficient information. Respond only with the classification and nothing else."
            
            # "You are a professional classifier in fact checking, your job is to be given claims with their evidences and classify them as one of the following categories: supported, refuted, or NEI. If you are unsure and don't have enough info, respond with NEI. Respond only with the classification and nothing else."
            # "You are a professional classifier in fact-checking. Your job is to receive claims along with their evidences, which now include both text and images, and classify them into one of the following categories: supported, refuted, or NEI (Not Enough Information). Respond with NEI only if you are unsure or lack sufficient information. Respond only with the classification and nothing else."
            
            # "You are a professional classifier in fact-checking. Your job is to receive claims along with their evidences, which now include both text and images, and classify them into one of the following categories: supported, refuted, or NEI (Not Enough Information). In cases where the image evidence contradicts the text evidence, you should prioritize the text evidence over the image. This means that the classification should be based primarily on the textual information provided. Respond with NEI only if you are unsure or lack sufficient information. Respond only with the classification and nothing else."
        },
        # # {
        # #     "role": "user",
        # #     "content": "claim: 'Hydroxychloroquine cures this 'virus. evidence: Some studies have found that hy-droxychloroquine could help alleviate symptoms associated with COVID-19, but the research is not conclusive. There is no cure or vaccine for the novel coronavirus. Health officials have not listed hy-droxychloroquine as a treatment for radiation sickness.",
        # # },
        # # {   "role": "assistant", "content": "refuted"},
        # # {
        # #     "role": "user",
        # #     "content": "claim: A specific species of frog can live for 200 years. evidence: There is limited research on the lifespan of this frog species."
        # # },
        # # {   "role": "assistant", "content": "NEI"},
        # # {
        # #     "role": "user",
        # #     "content": "claim: Donald Trump 'saved' the Veterans Day parade in New York City in 1995. evidence: Police estimated 500,000 people attended the largest military parade ever held in New York. Organizers, who placed the turnout at closer to a million, said the parade would not have been a success if it hadn't been for real estate developer Donald Trump, who contributed $200,000 and raised another $300,000."
        # # },
        # # {   "role": "assistant", "content": "supported"},
        
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
        {
            "role": "user",
            "content": prompt + "classify the claim based on given evidence",
            # if images_path is not None, then the image will be added here
        },
    ]
    if images_path:
        with open(
            "/mnt/d/Mocheg-main/Mocheg-main/data/test/resized_images/" + images_path,
            "rb",
        ) as image_file:
            # Convert image to Base64 encoding
            encoded_string = base64.b64encode(image_file.read()).decode()
            # Append the image in Base64 encoding to the user message
            messages[-1]["images"] = [encoded_string]

    data = {
        # "model": "llama2",
        # "model": "llama_fc",
        # "model": "llava_fc",
        # "model": "llava_fc_img",
        # "model": "llava_fc_img1",
        "model": "llava",
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
        # , "num_predict": 100
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
        return f"Error: {response.status_code} {response.text}"


if __name__ == "__main__":
    test_prompt = "decribe this image"
    # "claim: A U-Haul operated by antifa activists distributed 'shields, weapons & black bloc' to protesters in Louisville, Ky., and is linked to 'a George Soros-funded initiative. below is the image evidence"
    test_img_id = "100841-proof-27-b3ae38d2dc62a7e825d3d172d91b29f3.jpg"
    print(generate_response(test_prompt, test_img_id))
