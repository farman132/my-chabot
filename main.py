import os

import chainlit as cl

from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel

from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

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
    tracing_disabled=True,
)

agent = Agent(
    name="Farman Assistant",
    instructions="You are a helpfull assistant"
)

@cl.on_message
async def handle_message(message : cl.Message):
     result = await Runner.run(
        agent,
        input=message.content,
        run_config=config
     )
     await cl.Message(content=result.final_output).send()


