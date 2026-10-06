import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

questions = [
    "List 5 highly cited arXiv papers about universal adversarial perturbations. "
    "For each paper, give: title, first author, year, arXiv ID. "
    "Only list papers that really exist.",
    "List 5 arXiv papers about universal adversarial perturbations crafted in the "
    "latent space of the Stable Diffusion VAE. "
    "For each paper, give: title, first author, year, arXiv ID. "
    "Only list papers that really exist.",
]

for question in questions:
    print("=" * 70)
    print(question)
    print("-" * 70)
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=question,
        )
        print(response.text)
    except Exception as error:
        print("ÉCHEC :", error)