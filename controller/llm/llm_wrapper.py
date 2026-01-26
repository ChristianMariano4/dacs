from enum import Enum
import os
import time
import openai
from openai import Stream, ChatCompletion, OpenAI
import json

from controller.utils.constants import EVALUATION_LOG_PATH

LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"
GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
GPT4_O = "gpt-4o"
GPT_O4_MINI = "o4-mini"
GPT5 = "gpt-5" # The best model for coding and agentic tasks across domains
GPT5_MINI = "gpt-5-mini" # A faster, cost-efficient version of GPT-5 for well-defined tasks
GPT5_NANO = "gpt-5-nano" # Fastest, most cost-efficient version of GPT-5

PLAN_PROMPT_ID = "pmpt_68e90713b9408193b55cfa7573c17c370576d48f6ffbf9bf"
QUERY_PROMPT_ID = "pmpt_68e9237d54e8819588219a8d0b09e0ec048745458397c172"
SHORT_TERM_MEMORY_PROMPT_ID = "pmpt_68fb6f6eb20481959bf11be873e8ce7e03ae4d244586878c"
SAVE_TASK_FEEDBACK_PROMPT_ID = "pmpt_68e91e679d08819596f9fd50bbba4bb60783ed888cede905"
RETRIEVE_TASK_FEEDBACK_PROMPT_ID = "pmpt_691b7959f63c8197b22b544c7806e44600487163bee17f0a"
UPDATE_UNIVERSAL_FEEDBACK_PROMPT_ID = "pmpt_690dbf7c49a08197ba357393820e3a1a01e35afc9d9db34b"
CHOOSE_DIRECTION_PROMPT_ID = "pmpt_68e921121c3481959413d8ea3978f32a083d5502d67b3df6"
CREATE_FLYZONE_PROMPT_ID = "pmpt_68e9255a306c819784c286c70106af680a2d388474238928"
CREATE_GRAPH_PROMPT_ID = "pmpt_69047fda7b048195bd41c7f3ccba7f8f0a2d879dd1ddb53e"


PLAN_PROMPT_VERSION = "110"
QUERY_PROMPT_VERSION = "5"
SHORT_TERM_MEMORY_PROMPT_VERSION = "11"
SAVE_TASK_FEEDBACK_PROMPT_VERSION = "6"
RETRIEVE_TASK_FEEDBACK_PROMPT_VERSION = "8"
UPDATE_UNIVERSAL_FEEDBACK_PROMPT_VERSION = "12"
CHOOSE_DIRECTION_PROMPT_VERSION = "8"
CREATE_FLYZONE_PROMPT_VERSION = "30"
CREATE_GRAPH_PROMPT_VERSION = "13"


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "../assets/chat_log.txt")


class RequestType(Enum):
    PLAN = "plan"
    QUERY = "query"
    SHORT_TERM_MEMORY = "short_term_memory"

    FEEDBACK = "pmpt_68e91e679d08819596f9fd50bbba4bb60783ed888cede905"
    EXPLORE_DIRECTION = "pmpt_68e921121c3481959413d8ea3978f32a083d5502d67b3df6"
    FLYZONE = "pmpt_68e9255a306c819784c286c70106af680a2d388474238928"
    NEW_GRAPH = "pmpt_69047fda7b048195bd41c7f3ccba7f8f0a2d879dd1ddb53e"
    EVERGREEN_FEEDBACK = "pmpt_690dbf7c49a08197ba357393820e3a1a01e35afc9d9db34b"
    RETRIEVE_TASK_FEEDBACK = "pmpt_691b7959f63c8197b22b544c7806e44600487163bee17f0a"

class LLMWrapper:
    def __init__(self):
        
        self.llama_client = OpenAI(
            # base_url="http://10.66.41.78:8000/v1",
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        self.gpt_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def request(self, user_prompt: str = None, request_type: RequestType = RequestType.PLAN, image=None, images=None, model_name=GPT5, stream=False, variables: tuple = None) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
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
                                "text": user_prompt,
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
                        "version": PLAN_PROMPT_VERSION
                    },
                    input=input_payload,
                    stream=stream
                )

            case RequestType.FEEDBACK:
                response = client.responses.create(
                    prompt={
                        "id": SAVE_TASK_FEEDBACK_PROMPT_ID,
                        "version": SAVE_TASK_FEEDBACK_PROMPT_VERSION
                    },
                    input=user_prompt,
                    stream=stream
                )

            case RequestType.RETRIEVE_TASK_FEEDBACK:
                response = client.responses.create(
                    prompt={
                        "id": RETRIEVE_TASK_FEEDBACK_PROMPT_ID,
                        "version": RETRIEVE_TASK_FEEDBACK_PROMPT_VERSION,
                        "variables": {
                            "user_request": variables[0],
                            "candidate_preferences_json": json.dumps(variables[1])
                        }
                    },
                )


            case RequestType.EVERGREEN_FEEDBACK:
                response = client.responses.create(
                    prompt={
                        "id": UPDATE_UNIVERSAL_FEEDBACK_PROMPT_ID,
                        "version": UPDATE_UNIVERSAL_FEEDBACK_PROMPT_VERSION
                    },
                    input=user_prompt,
                    stream=stream
                )

            case RequestType.SHORT_TERM_MEMORY:
                    response = client.responses.create(
                    prompt={
                        "id": SHORT_TERM_MEMORY_PROMPT_ID,
                        "version": SHORT_TERM_MEMORY_PROMPT_VERSION
                    },
                    input=user_prompt,
                    stream=stream
                )

            case RequestType.FLYZONE:
                input_payload = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_prompt + "\n\nPlease respond in JSON format with fields: `points_list` and `reason`.",
                            }
                        ],
                    }
                ]

                if image != None:
                    input_payload[0]["content"].append({"type": "input_text", "text": f"Image:"})
                    input_payload[0]["content"].append({"type": "input_image", "image_url": "data:image/jpeg;base64," + image})
                
                response = client.responses.create(
                    prompt={
                        "id": CREATE_FLYZONE_PROMPT_ID,
                        "version": CREATE_FLYZONE_PROMPT_VERSION
                    },
                    input=input_payload,
                    stream=stream
                )

            case RequestType.QUERY:
                assert image is not None, f"Image not given in a {RequestType.QUERY} request"
                response = client.responses.create(
                    prompt={
                        "id": QUERY_PROMPT_ID,
                        "version": QUERY_PROMPT_VERSION
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
                        {"type": "input_text", "text": f"Yaw: {direction}"}
                    )
                    input_payload[0]["content"].append(
                        {"type": "input_image", "image_url": "data:image/jpeg;base64," + img_b64}
                    )

                response = client.responses.create(
                    prompt={
                        "id": CHOOSE_DIRECTION_PROMPT_ID,
                        "version": CHOOSE_DIRECTION_PROMPT_VERSION
                    },
                    input=input_payload,
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
                        "id": CREATE_GRAPH_PROMPT_ID,
                        "version": CREATE_GRAPH_PROMPT_VERSION
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
            if user_prompt:
                f.write(user_prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response
        # Access the raw text
        raw_text = response.output[1].content[0].text
        # print("Raw text:", raw_text)

        # Parse it:
        parsed = json.loads(raw_text)

        # Evaluation log - received LLM plan
        with open(EVALUATION_LOG_PATH, "a") as f:
            f.write(f"[{time.time()}] Plan from LLM: {parsed}\n")

        return parsed