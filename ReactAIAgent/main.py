import json
import os
import re

from openai import OpenAI

from Simple_test.ReactAIAgent.agent import CustomerServiceAgent
from Simple_test.ReactAIAgent.tools.calc import calculate
from Simple_test.ReactAIAgent.tools.query_by_product_data import query_by_product_name
from Simple_test.ReactAIAgent.tools.read_promotions import read_store_promotions

from op_llm_client import OllamaClient

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config
def get_client(config):
    if config['openai'].get('use_model', True):
        return OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
    else:
        return OllamaClient()
def get_max_iterations(config):
    if config['ollama']['use_model']:
        return config['ollama']['max_iterations']
    elif config['openai']['use_model']:
        return config['openai']['max_iterations']
    else:
        return 10
def main():
    config = load_config()
    try:
        client = get_client(config)
        agent = CustomerServiceAgent(client, config)
    except Exception as e:
        print(f"Error initializing the AI client: {str(e)}")
        print("Please check your configuration and ensure the AI service is running.")
        return
    tools = {
        "query_by_product_name": query_by_product_name,
        "read_store_promotions": read_store_promotions,
        "calculate": calculate,
    }

    while True:
        query = input("输入的问题或输入 '退出' 来结束: ")
        if query.lower() == '退出':
            break
        iteration = 0
        max_iterations = get_max_iterations(config)
        while iteration < max_iterations:
            try:
                result = agent(query)
                action_re = re.compile('^Action: (\w+): (.*)$')
                actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
                if actions:
                    action_parts = result.split("Action:", 1)[1].strip().split(": ", 1)
                    tool_name = action_parts[0]
                    tool_args = action_parts[1] if len(action_parts) > 1 else ""
                    if tool_name in tools:
                        # print(f"执行工具：{tool_name}", tool_args)
                        try:
                            observation = tools[tool_name](tool_args)
                            query = f"Observation: {observation}"
                        except Exception as e:
                            query = f"Observation: Error occurred while executing the tool: {str(e)}"
                    else:
                        query = f"Observation: Tool '{tool_name}' not found"
                elif "Answer:" in result:
                    print(f"客服回复：{result.split('Answer:', 1)[1].strip()}")
                    break
                else:
                    query = "Observation: No valid action or answer found. Please provide a clear action or answer."
            except Exception as e:
                print(f"An error occurred while processing the query: {str(e)}")
                print("Please check your configuration and ensure the AI service is running.")
                break
            iteration += 1
            if iteration == max_iterations:
                print("Reached maximum number of iterations without a final answer.")




if __name__ == '__main__':
    # 你们家卖那些球？
    main()