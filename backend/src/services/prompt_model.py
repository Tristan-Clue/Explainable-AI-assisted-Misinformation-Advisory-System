from google import genai
import ollama

from config import OLLAMA_BASE_URL, GEMINI_API_KEY

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    
try:
    client = ollama.Client(host=OLLAMA_BASE_URL)
except Exception as e:
    print(f"Error initializing Ollama client: {e}")

ollamalist = ["phi3", "llama3.1", "deepseek-r1", "gemma4:e2b", "llama3.2:3b"]
geminilist = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
allmodels = ollamalist + geminilist


def prompt_model(model: str, prompt: str, temperature=None) -> str:
    try:
        if model not in allmodels:
            raise ValueError(f"Model {model} not found in available models.")

        if model.startswith("gemini"):
            print(f"Asking model: {model}")
            config = {}
            if temperature is not None:
                config["temperature"] = temperature
            response = client.models.generate_content(
                model=model, contents=prompt, config=config
            ).text
        else:
            print(f"Asking model: {model} at URL {OLLAMA_BASE_URL}")
            options = {}
            if temperature is not None:
                options["temperature"] = temperature
            response = client.generate(
                model=model,
                prompt=prompt,
                stream=False,
                options=options,
            )["response"]
        return response

    except Exception as e:
        print(f"An error occurred while generating response {e}")
        return f"Unable to generate response: {e}"
