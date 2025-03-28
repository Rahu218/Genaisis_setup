# Need to add VERTEXAI_EMBEDDING_MODEL in the environment variables
from GenAIsis.Components.EmbeddingsComponents import VertexAIEmbeddingsComponent, AzureOpenAIEmbeddingsComponent


def create_embeddings(input_str):
    # Generate embeddings using VertexAIEmbeddingsComponent
    embeddings = VertexAIEmbeddingsComponent().build()

    # Generate embeddings using AzureOpenAIEmbeddingsComponent
    #azur_embeddings = AzureOpenAIEmbeddingsComponent().build()

    result = embeddings.embed_query(input_str)        
    return result

text = "How to Train a Neural Network in low end Hardware? THis is a sample to test the embeddings functionality"
print(create_embeddings(text))