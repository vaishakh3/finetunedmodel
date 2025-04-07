from fastapi import FastAPI, WebSocket
from agent import Agent
import json

app = FastAPI()

SYSTEM_PROMPT = """You are an empathetic assistant speaking to a distressed caller in a crisis.
Do not use greetings like "911, what's your emergency?". Respond calmly, compassionately, and directly to help the person in need."""


@app.websocket("/llm")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent = Agent(system_prompt=SYSTEM_PROMPT)

    while True:
        data = await websocket.receive_text()
        hume_socket_message = json.loads(data)

        message, chat_history = agent.parse_hume_message(hume_socket_message)

        print("User message:", message)
        print("Chat history:", chat_history)
        print("Detected emotions:", agent.latest_emotions)

        responses = agent.get_responses(message, chat_history)

        print("Responses:", responses)

        for response in responses:
            await websocket.send_text(response)
