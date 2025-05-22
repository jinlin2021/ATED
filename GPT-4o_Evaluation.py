import base64
import time
import json
import os
from openai import OpenAI

# === Set Open AI API ===
client = OpenAI(
    base_url=" ",  # Replace with your API URL
    api_key= "sk-XXXXX" # Replace with your API key
)

# === GPT prompt format ===
GPT_JUDGE_PROMPT = '''
You are an AI designed to evaluate and score the performance of two AI assistants in describing a given image. Your primary focus is on the accuracy and detailedness of their descriptions. You will assess the accuracy by checking for hallucinations—any part of the description that is inconsistent with the image content. For detailedness, you will consider how rich the response is in necessary details, excluding any hallucinated parts. You will provide scores on a scale from 1 to 10 for each assistant separately, based on these criteria. After scoring, you will offer an explanation for your evaluation, ensuring it is free from bias and not influenced by the order of presentation of the responses.

Input format:

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy:
Scores of the two answers:

Detailedness:
Scores of the two answers:

'''

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def load_captions(jsonl_path):
    result = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            image_id = str(data["image_id"])
            result[image_id] = data["caption"]
    return result



# === Set your path ===
image_folder = "YOUR IMAGE PATH"  # Your image path, your image called <image_id>.jpg
file1 = "MODEL 1 OUTPUT PATH" # Your first model output path
file2 = "MODEL 2 OUTPUT PATH" # Your second model output path

captions1 = load_captions(file1)
captions2 = load_captions(file2)

# === Get the id in both group ===
common_ids = set(captions1.keys()) & set(captions2.keys())

# === Evaluation ===
results = []

for image_id in sorted(common_ids):
    image_path = os.path.join(image_folder, f"{int(image_id):03d}.jpg")


    base64_image = encode_image(image_path)

    # Create prompt
    ans1 = captions1[image_id]
    ans2 = captions2[image_id]
    prompt = GPT_JUDGE_PROMPT.format(ans1, ans2)

    # Call GPT-4o model

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20", # Replace with your model name
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=1
    )

    content = response.choices[0].message.content
    print(f"✅ Image ID: {image_id} Evaluation completed.")
    print(content)
    results.append({
        "image_id": image_id,
        "result": content
    })
    time.sleep(1)


# === Save your result===
output_path = "XXXXXXXX.jsonl" # Replace with your output path
with open(output_path, "w", encoding="utf-8") as f:
    for entry in results:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\nAll evaluation is finished, the result is saved to {output_path}")