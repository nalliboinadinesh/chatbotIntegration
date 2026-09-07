import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY. Add it to your environment or .env file.")

client = genai.Client(api_key=api_key)


class AIRequest(BaseModel):
    system_prompt: str
    content: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "chatbot-integration"}


@app.post("/api/ai/generate")
def generate_content(request: AIRequest):
    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            temperature=0.4,
            max_output_tokens=1000,
        ),
        contents=request.content,
    )
    print(response.text)

    return {"content": response.text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
