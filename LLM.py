import os
import json
import base64
import re
import time
from pathlib import Path
from PIL import Image
from utils import resize_and_convert_to_base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def generate_response_openai(image_path, prompt):
    
    image_b64 = resize_and_convert_to_base64(image_path= image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_b64}}
            ]}
        ],
        max_tokens=1024
    )
    return response
    