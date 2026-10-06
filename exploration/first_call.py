import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain in two sentences what a universal adversarial perturbation is.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    ),
)
print(response.text)

usage = response.usage_metadata
print("Tokens du prompt     :", usage.prompt_token_count)
print("Tokens de la réponse :", usage.candidates_token_count)
print("Tokens de réflexion  :", usage.thoughts_token_count)
print("Total                :", usage.total_token_count)