from enum import Enum
import os
import openai
from openai import Stream, ChatCompletion, OpenAI

LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"
GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
GPT4_O = "gpt-4o"
GPT_O4_MINI = "o4-mini"
GPT5 = "gpt-5" # The best model for coding and agentic tasks across domains
GPT5_MINI = "gpt-5-mini" # A faster, cost-efficient version of GPT-5 for well-defined tasks
GPT5_NANO = "gpt-5-nano" # Fastest, most cost-efficient version of GPT-5

PLAN_PROMPT_ID = "pmpt_68e90713b9408193b55cfa7573c17c370576d48f6ffbf9bf"
FEEDBACK_PROMPT_ID = "pmpt_68e91e679d08819596f9fd50bbba4bb60783ed888cede905"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "../assets/chat_log.txt")


class RequestType(Enum):
    PLAN = "plan"
    FEEDBACK = "feedback"
    EXPLORE_DIRECTION = "explore_direction"
    PROBE = "probe"
    LIGHT = "light"

class LLMWrapper:
    def __init__(self, temperature=1):
        self.temperature = temperature
        
        self.llama_client = OpenAI(
            # base_url="http://10.66.41.78:8000/v1",
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        self.gpt_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def request(self, user_prompt, image=None, images=None, model_name=GPT5, stream=False, request_type: RequestType = RequestType.SIMPLE) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        if model_name == LLAMA3:
            client = self.llama_client
        else:
            client = self.gpt_client
        
        match request_type:
            case RequestType.PLAN:
                response = client.responses.create(
                    prompt={
                        "id": PLAN_PROMPT_ID,
                        "version": "2"
                    },
                    input=user_prompt,
                    stream=stream
                )
            case RequestType.FEEDBACK:
                response = client.responses.create(
                    prompt={
                        "id": FEEDBACK_PROMPT_ID,
                        "version": "1"
                    },
                    input=user_prompt,
                    stream=stream
                )

            case RequestType.PROBE:
                assert image is not None, f"Image not given in a {RequestType.PROBE} request"
                response = client.responses.create(
                    prompt={
                        "id": FEEDBACK_PROMPT_ID,
                        "version": "1"
                    },
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": image}}
                            ]
                        }
                    ],
                    stream=stream
                )
            
            case RequestType.EXPLORE_DIRECTION:
                assert images is not None, f"Images not given in a {RequestType.EXPLORE_DIRECTION} request"
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[0], "name": "north.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[1], "name": "north-east.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[2], "name": "east.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[3], "name": "south-east.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[4], "name": "south.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[5], "name": "south-west.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[6], "name": "west.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[7], "name": "north-west.jpg"}},
                            ]
                        }
                    ],
                    temperature=self.temperature,
                    stream=stream,
                    response_format={"type": "json_object"} 
                )


        # save the message in a txt
        with open(chat_log_path, "a") as f:
            f.write(user_prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response
        # Access the raw text
        raw_text = response.output[1].content[0].text
        # print("Raw text:", raw_text)

        # If it's JSON (as in your example), parse it:
        import json
        parsed = json.loads(raw_text)

        return parsed