import json

import requests


class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434"):
        self.base_url = base_url

    def chat_completions_create(self, model, messages, temperature=0.7):
        url = f"{self.base_url}/api/generate"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": self._format_message(messages),
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }
        try:
            response = requests.post(url=url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                response_text = response.text
                data = json.loads(response_text)
                actual_response = data['response']
                return actual_response
            else:
                raise Exception(f"Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to Ollama server at {self.base_url}. "
                              f"Make sure Ollama is running and accessible.")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to Ollama server at {self.base_url} timed out. "
                           f"The server might be overloaded or not responding.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"The specified model might not be available on the Ollama server. "
                                 f"Error: {str(e)}")
            else:
                raise

    def _format_message(self, messages):
        format_prompt = ""
        for message in messages:
            if message["role"] == "system":
                format_prompt += f"System: {message['content']}\n"
            elif message["role"] == "user":
                format_prompt += f"Human: {message['content']}\n"
            elif message["role"] == "assistant":
                format_prompt += f"Assistant: {message['content']}\n"
        return format_prompt.strip()