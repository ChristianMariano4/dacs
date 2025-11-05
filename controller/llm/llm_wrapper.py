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
SHORT_MEMORY_PROMPT_ID = "pmpt_68fb6f6eb20481959bf11be873e8ce7e03ae4d244586878c"
PROBE_PROMPT_ID = "pmpt_68e9237d54e8819588219a8d0b09e0ec048745458397c172"
DIRECTION_PROMPT_ID = "pmpt_68e921121c3481959413d8ea3978f32a083d5502d67b3df6"
FLYZONE_PROMPT_ID = "pmpt_68e9255a306c819784c286c70106af680a2d388474238928"
NEW_GRAPH_PROMPT_ID = "pmpt_69047fda7b048195bd41c7f3ccba7f8f0a2d879dd1ddb53e"


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "../assets/chat_log.txt")


class RequestType(Enum):
    PLAN = "pmpt_68e90713b9408193b55cfa7573c17c370576d48f6ffbf9bf"
    FEEDBACK = "pmpt_68e91e679d08819596f9fd50bbba4bb60783ed888cede905"
    SHORT_MEMORY = "pmpt_68fb6f6eb20481959bf11be873e8ce7e03ae4d244586878c"
    EXPLORE_DIRECTION = "pmpt_68e921121c3481959413d8ea3978f32a083d5502d67b3df6"
    PROBE = "pmpt_68e9237d54e8819588219a8d0b09e0ec048745458397c172"
    FLYZONE = "pmpt_68e9255a306c819784c286c70106af680a2d388474238928"
    NEW_GRAPH = "pmpt_69047fda7b048195bd41c7f3ccba7f8f0a2d879dd1ddb53e"

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

    def request(self, user_prompt, request_type: RequestType, image=None, images=None, model_name=GPT5, stream=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        if model_name == LLAMA3:
            client = self.llama_client
        else:
            client = self.gpt_client
        
        match request_type:
            case RequestType.PLAN:
                input_payload = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    user_prompt
                                    + "\n\nPlease respond in JSON format with fields: "
                                    "direction, distance, region_name."
                                ),
                            }
                        ],
                    }
                ]

                if image != None:
                    input_payload[0]["content"].append({"type": "input_text", "text": f"Image:"})
                    input_payload[0]["content"].append({"type": "input_image", "image_url": "data:image/jpeg;base64," + image})
                    
                response = client.responses.create(
                    prompt={
                        "id": PLAN_PROMPT_ID,
                        "version": "14"
                    },
                    input=input_payload,
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

            case RequestType.SHORT_MEMORY:
                    response = client.responses.create(
                    prompt={
                        "id": SHORT_MEMORY_PROMPT_ID,
                        "version": "6"
                    },
                    input=user_prompt,
                    stream=stream
                )

            case RequestType.FLYZONE:
                response = client.responses.create(
                    prompt={
                        "id": FLYZONE_PROMPT_ID,
                        "version": "2"
                    },
                    input=user_prompt + "\n\nPlease respond in JSON format with fields: points_list.",
                    stream=stream
                )

            case RequestType.PROBE:
                assert image is not None, f"Image not given in a {RequestType.PROBE} request"
                response = client.responses.create(
                    prompt={
                        "id": PROBE_PROMPT_ID,
                        "version": "2"
                    },
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_prompt},
                                {"type": "input_image", "image_url": "data:image/jpeg;base64," + image}
                            ]
                        }
                    ],
                    stream=stream
                )
            
            case RequestType.EXPLORE_DIRECTION:
                assert images is not None, f"Images not given in a {RequestType.EXPLORE_DIRECTION} request"
                input_payload = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    user_prompt
                                    + "\n\nPlease respond in JSON format with fields: "
                                    "direction, distance, region_name."
                                ),
                            }
                        ],
                    }
                ]

                # Iterate over dict items and append direction + image pairs
                for direction, img_b64 in images.items():
                    input_payload[0]["content"].append(
                        {"type": "input_text", "text": f"Direction: {direction}"}
                    )
                    input_payload[0]["content"].append(
                        {"type": "input_image", "image_url": "data:image/jpeg;base64," + img_b64}
                    )

                response = client.responses.create(
                    prompt={
                        "id": DIRECTION_PROMPT_ID,
                        "version": "3"
                    },
                    input=[
                        {
                            "role": "user",
                            "content": input_payload
                        }
                    ],
                    stream=stream,
                )
            
            case RequestType.NEW_GRAPH:
                input_payload = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    user_prompt
                                ),
                            }
                        ],
                    }
                ]
                if image != None:
                    input_payload[0]["content"].append({"type": "input_text", "text": f"Image:"})
                    input_payload[0]["content"].append({"type": "input_image", "image_url": "data:image/jpeg;base64," + image})

                response = client.responses.create(
                    prompt={
                        "id": NEW_GRAPH_PROMPT_ID,
                        "version": "2"
                    },
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_prompt},

                            ]
                        }
                    ],
                    stream=stream
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

        # Parse it:
        import json
        parsed = json.loads(raw_text)

        return parsed