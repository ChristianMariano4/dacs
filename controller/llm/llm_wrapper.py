from enum import Enum
import os
import time
from openai import Stream, ChatCompletion, OpenAI
import json
import requests


from controller.utils.constants import EVALUATION_LOG_PATH, SYSTEM_PROMPT_PATHS

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
    SAVE_TASK_FEEDBACK = "save_task_feedback"
    CHOOSE_DIRECTION = "pmpt_68e921121c3481959413d8ea3978f32a083d5502d67b3df6"
    CREATE_FLYZONE = "pmpt_68e9255a306c819784c286c70106af680a2d388474238928"
    CREATE_GRAPH = "pmpt_69047fda7b048195bd41c7f3ccba7f8f0a2d879dd1ddb53e"
    UNIVERSAL_FEEDBACK = "pmpt_690dbf7c49a08197ba357393820e3a1a01e35afc9d9db34b"
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

    def _load_system_prompt(self, request_type: RequestType) -> str:
        """
        Load system prompt from file based on request type.
        
        How it works:
        1. Gets the RequestType name (e.g., RequestType.PLAN -> "PLAN")
        2. Looks up the file path in SYSTEM_PROMPT_PATHS dictionary
        3. Reads and returns the file contents
        
        Returns empty string if file not found (allows graceful degradation).
        """
        prompt_path = SYSTEM_PROMPT_PATHS.get(request_type.name)
        
        if not prompt_path:
            print(f"Warning: No prompt path configured for {request_type.name}")
            return ""
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: System prompt file not found at {prompt_path}")
            return ""

    def _openai_response_api_request(self, user_prompt: str = None, request_type: RequestType = RequestType.PLAN, image=None, images=None, stream=False, variables: tuple = None) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
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

            case RequestType.SAVE_TASK_FEEDBACK:
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
                    },
                    input=user_prompt,
                    stream=stream                )


            case RequestType.UNIVERSAL_FEEDBACK:
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

            case RequestType.CREATE_FLYZONE:
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
            
            case RequestType.CHOOSE_DIRECTION:
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
            
            case RequestType.CREATE_GRAPH:
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

    def _ollama_request(
        self,
        user_prompt: str = None, 
        system_prompt: str = None,
        image=None,
        images=None,
        model_name: str = "kimi-k2:1t-cloud",
        stream: bool = False
    ) -> dict | str:
        
        # Step 1: Build messages array
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Step 2: Build the user message (this is where images belong!)
        user_message = {
            "role": "user",
            "content": user_prompt or ""
        }
        
        # Attach images TO THE USER MESSAGE, not to the payload root
        if image is not None:
            user_message["images"] = [image]
        elif images is not None:
            user_message["images"] = list(images.values())
        
        messages.append(user_message)
        
        # Step 3: Build request payload (no images here!)
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": stream
        }
        
        # Step 4: Send request
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload
        )
        
        # Step 5: Parse response
        raw_content = response.json()["message"]["content"]
        
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            return raw_content
    
    def request(
        self,
        user_prompt: str = None,
        request_type: RequestType = RequestType.PLAN,
        image=None,
        images=None,
        stream=False,
        variables: tuple = None,
        use_ollama: bool = False,
        ollama_model: str = "kimi-k2.5:cloud"
    ) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        """
        Unified request method that routes to either OpenAI or Ollama.
        
        Args:
            user_prompt: The text prompt to send
            request_type: Type of request (used by OpenAI for prompt selection)
            image: Single base64-encoded image (optional)
            images: Dictionary of images keyed by direction (optional)
            stream: Whether to stream the response
            variables: Tuple of variables for certain request types
            use_ollama: If True, use Ollama; if False, use OpenAI
            ollama_model: Model name when using Ollama
            
        Returns:
            Parsed JSON response (dict) or stream object if stream=True
        """

        response = None
        if use_ollama:
            # Ollama path: simpler, just sends the prompt directly
            system_prompt = self._load_system_prompt(request_type)
            response = self._ollama_request(
                user_prompt=user_prompt + "\n(output a json object, but not use ```json)",
                system_prompt=system_prompt,
                image=image,
                images=images,
                model_name=ollama_model,
                stream=stream
            )
            if response is not None:
                # save the message in a txt
                with open(chat_log_path, "a") as f:
                    if system_prompt and user_prompt:
                        f.write(system_prompt + "\n---\n")
                        f.write(user_prompt + "\n---\n")
                    if not stream:
                        f.write(json.dumps(response) + "\n---\n")

        else:
            # OpenAI path: uses the stored prompt templates
            response = self._openai_response_api_request(
                user_prompt=user_prompt,
                request_type=request_type,
                image=image,
                images=images,
                stream=stream,
                variables=variables
            )
            if response is not None:
                # save the message in a txt
                with open(chat_log_path, "a") as f:
                    if user_prompt:
                        f.write(user_prompt + "\n---\n")
                    if not stream:
                        f.write(json.dumps(response) + "\n---\n")
        

        return response