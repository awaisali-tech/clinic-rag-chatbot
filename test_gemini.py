from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")
print(f"Key loaded: {api_key[:10]}...")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)

print(response.choices[0].message.content)