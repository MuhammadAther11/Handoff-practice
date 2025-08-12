from pydantic import BaseModel
from agents import (
    Agent, 
    handoff,
    Runner,
    RunConfig ,AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig
)
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")


# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


urdu_agent = Agent(
    name="Urdu agent",
    instructions="You only speak Urdu."
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English"
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[urdu_agent, english_agent],
)


async def main(input: str):
    result = await Runner.run(triage_agent, input=input, run_config=config)
    print(result.final_output)

if __name__ == "__main__":
        asyncio.run(main("السلام عليكم"))
        