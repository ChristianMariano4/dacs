from enum import Enum
import os
import openai
from openai import Stream, ChatCompletion

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4o"
LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

class RequestType(Enum):
    EXPLORE_DIRECTION = "explore_direction"
    SIMPLE = "simple"
    SINGLE_IMAGE = "single_image"

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.llama_client = openai.OpenAI(
            # base_url="http://10.66.41.78:8000/v1",
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        self.gpt_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def request(self, prompt, image=None, images=None, model_name=GPT4, stream=False, request_type: RequestType = RequestType.SIMPLE) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        if model_name == LLAMA3:
            client = self.llama_client
        else:
            client = self.gpt_client
        
        match request_type:
            case RequestType.SIMPLE:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=stream,
                )
            
            case RequestType.SINGLE_IMAGE:
                assert image is not None, f"Image not given in a {RequestType.SINGLE_IMAGE} request"
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}, prompt]}],
                    temperature=self.temperature,
                    stream=stream,
                )
            
            case RequestType.EXPLORE_DIRECTION:
                assert images is not None, f"Images not given in a {RequestType.SINGLE_IMAGE} request"
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image[0], "name": "north.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image[1], "name": "north-east.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image[2], "name": "east.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image[3], "name": "south-east.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image[4], "name": "south.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image[5], "name": "south-west.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image[6], "name": "west.jpg"}},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image[7], "name": "north-west.jpg"}},
                            ]
                        }
                    ],
                    temperature=self.temperature,
                    stream=stream,
                    response_format={"type": "json_object"} 
                )


        # save the message in a txt
        with open(chat_log_path, "a") as f:
            f.write(prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response

        return response.choices[0].message.content