from agents import (
    Agent, 
    handoff,
    Runner,
    RunConfig ,AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper
)
from dotenv import load_dotenv
import os
import asyncio
import rich

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

# 1 example

# billing_agent = Agent(name="Billing agent")
# refund_agent = Agent(name="Refund agent")


# triage_agent = Agent(name="Triage agent", handoffs=[billing_agent, handoff(refund_agent)])

# async def main(input: str):
#     result = await Runner.run(triage_agent, input=input, run_config=config)
#     print(result.final_output)

# if __name__ == "__main__":
#     input_text = "I want a bill for my last purchase."
#     asyncio.run(main(input_text))

# 2 example


# billing_agent = Agent(name="Billing agent", instructions="Handle billing questions.")
# refund_agent  = Agent(name="Refund agent",  instructions="Handle refunds.")

# triage_agent = Agent(
#     name="Triage agent",
#     instructions=(
#         "Help the user with their questions. "
#         "If they ask about billing, handoff to the Billing agent. "
#         "If they ask about refunds, handoff to the Refund agent."
#     ),
#     handoffs=[billing_agent, handoff(refund_agent)],  # either direct agent or `handoff(...)`
# )

# async def main():
#     result = await Runner.run(triage_agent, "I need to check refund status." , run_config=config)
#     print(result.final_output)
#     # print(result.new_items)
#     # print(result._last_agent)


# if __name__ == "__main__":
#     asyncio.run(main())

# 3 example

next_js_assistant = Agent(
     name="Next.js Assistant",
        instructions=(
            "You are a Next.js expert. "
            "Answer questions about Next.js and web development."
        )
)

python_assistant = Agent(
    name="Python Assistant",
    instructions=(
        "You are a Python expert. "
        "Answer questions about Python programming."
    )
)

python_handoff = handoff(
     agent=python_assistant,
     tool_name_override='specialized_python_agent',
     tool_description_override="you are specialized in Python programming.",
)

lead_agent = Agent(
    name="Lead Agent", 
    instructions="You are Lead agent you will be given task and you have to handoff to the specialized agents accordingly",
    handoffs=[
        next_js_assistant,
        python_handoff
        ]
)

result =Runner.run_sync(
     starting_agent=lead_agent,
    input="I am having some issue with python decorators",
    run_config=config
    
)

print("last Agent>>>>>",result.last_agent)   
rich.print("result>>>>", result.final_output)