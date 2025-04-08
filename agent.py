import json
import re
import inflect
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from google import genai
from google.genai import types


def convert_chat_history(chat_history, system_prompt=None):
    gemini_messages = []

    if system_prompt:
        gemini_messages.append(
            types.Content(role="user", parts=[types.Part(text=system_prompt)])
        )

    for msg in chat_history:
        if isinstance(msg, SystemMessage):
            continue

        role = "user" if isinstance(msg, HumanMessage) else "model"

        if isinstance(msg.content, str):
            gemini_messages.append(
                types.Content(role=role, parts=[types.Part(text=msg.content)])
            )

    return gemini_messages


class Agent:
    def __init__(self, *, system_prompt: str):
        self.system_prompt = system_prompt
        self.latest_emotions = []

        # Auth automatically picks up ADC
        self.client = genai.Client(
            vertexai=True,
            project="965267089646",
            location="us-central1",
        )

        self.model_name = "projects/965267089646/locations/us-central1/endpoints/3860118143795986432"

        self.generation_config = types.GenerateContentConfig(
            temperature=0.9,
            top_p=0.95,
            max_output_tokens=1024,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
        )

    def add_prosody_to_utterance(self, utterance: str, prosody: dict) -> str:
        if prosody:
            prosody_string = ", ".join(prosody.keys())
            return f"{utterance} [{prosody_string}]"
        return utterance

    def parse_hume_message(self, messages_payload: dict) -> [str, list]:
        messages = messages_payload["messages"]
        last_user_message = messages[-1]["message"]["content"]
        chat_history = [SystemMessage(content=self.system_prompt)]

        all_emotions = []

        for message in messages[:-1]:
            message_object = message["message"]
            role = message_object["role"]
            content = message_object["content"]

            prosody_scores = message.get("models", {}).get("prosody", {}).get("scores", {})
            sorted_entries = sorted(prosody_scores.items(), key=lambda x: x[1], reverse=True)
            top_entries = {k: v for k, v in sorted_entries[:3]}
            all_emotions.extend(list(top_entries.keys()))

            contextualized_utterance = self.add_prosody_to_utterance(content, top_entries)

            if role == "user":
                chat_history.append(HumanMessage(content=contextualized_utterance))
            elif role == "assistant":
                chat_history.append(AIMessage(content=contextualized_utterance))

        self.latest_emotions = list(set(all_emotions))
        return [last_user_message, chat_history]

    def get_responses(self, message: str, chat_history=None) -> list[str]:
        if chat_history is None:
            chat_history = []

        emotions_str = ", ".join(self.latest_emotions) if self.latest_emotions else "none detected"

        empathetic_prompt = f"""
The caller just said: "{message}"
They are expressing the following emotions: {emotions_str}

Respond as an empathetic assistant. Be calm, reassuring, and helpful.
Avoid greetings like "911, what's your emergency?" and get straight to asking:
- The location of the emergency
- The nature of the incident
- Whether anyone is hurt

Keep your tone human-like and compassionate.
        """.strip()

        chat_history.append(HumanMessage(content=empathetic_prompt))
        input_messages = convert_chat_history(chat_history)

        output = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=input_messages,
            config=self.generation_config,
        ):
            output += chunk.text

        output = re.sub(r"(?i)^911[, ]+what'?s your emergency[\?]?[ ]*", "", output).strip()

        numbers = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", output)
        for number in numbers:
            output = output.replace(number, self.number_to_words(number), 1)

        return [
            json.dumps({"type": "assistant_input", "text": output}),
            json.dumps({"type": "assistant_end"})
        ]

    def number_to_words(self, number: str) -> str:
        p = inflect.engine()
        return p.number_to_words(number)
