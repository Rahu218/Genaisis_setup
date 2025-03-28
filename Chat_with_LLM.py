from GenAIsis import ChatLLMBuilder, ChatLLMWithMemoryBuilder
from GenAIsis.Components.ModelComponents import ChatVertexAIComponent, AzureChatOpenAIComponent
import dotenv
import os

from dotenv import dotenv_values

# config = dotenv_values(".env")  # Loads the .env file as a dictionary
# print(config)

dotenv.load_dotenv('.env')

print(os.getenv("OPENAI_API_KEY"))

# Example code to chart with LLM (OpenAI Models)

# For the below function to work, .env file should be created and updated with required fields and values
def chat_with_openaimodels():
    
    # Option 1: Using ChatLLMBuilder
    template = "Complete the sentences:\n {sentence1} \n {sentence2}"

    builder = ChatLLMBuilder()
    response = (builder.set_prompt(template, sentence1="The sky is", sentence2="Grass is", ).
            llm_chat.chat_azure_openai(max_tokens=200, temperature=0.5).build())
    # You can tweak the hyperparameters like max_tokens and temperature as per your requirement

    print(response)

    # Option 2: Directly using AzureChatOpenAIComponent
    model = AzureChatOpenAIComponent()
    prompt = "How to Train a Neural Network in low end Hardware?"

    response = model.build(prompt, max_tokens=200, temperature=0.5)
    # You can tweak the hyperparameters like max_tokens and temperature as per your requirement

    print(response)

def chat_with_googlemodels():
    # Option 1: Using ChatLLMBuilder
    template = "Complete the sentences:\n {sentence1} \n {sentence2}"

    builder = ChatLLMBuilder()
    response = (builder.set_prompt(template, sentence1="The sky is", sentence2="Grass is", ).
            llm_chat.chat_google_vertexai(max_tokens=200, temperature=0.5).build())
    # You can tweak the hyperparameters like max_tokens and temperature as per your requirement

    print(response)

    # Option 2: Directly using ChatVertexAIComponent
    model = ChatVertexAIComponent()
    prompt = "How to Train a Neural Network in low end Hardware?"

    response = model.build(prompt, max_tokens=200, temperature=0.5)
    # You can tweak the hyperparameters like max_tokens and temperature as per your requirement

    print(response)


if __name__ == "__main__":
    prompt = "How to Train a Neural Network in low end Hardware?"
    chat_with_openaimodels()
    chat_with_googlemodels()