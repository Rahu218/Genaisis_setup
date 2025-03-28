# Chat with LLM with memory

import os
from dotenv import load_dotenv
from GenAIsis import ChatLLMBuilder, ChatLLMWithMemoryBuilder

load_dotenv('.env')


def chat_with_memory_openaimodels():
    builder = ChatLLMWithMemoryBuilder()
    response = (builder.set_question("Get java code for to check if it's a string or int").
            llm_chat.chat_azure_openai().build())

    print(response)

    response2 = (builder.set_question("What was my last prompt").
            llm_chat.chat_azure_openai().build())

    print(response2)

if __name__ == "__main__":
    chat_with_memory_openaimodels()
    #chat_with_memory_googlemodels()