from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re

# Importing Genaisis
from GenAIsis.Components.ModelComponents import ChatVertexAIComponent, AzureChatOpenAIComponent


app = FastAPI()

# Azure OpenAI Client Setup
client = AzureOpenAI(
    api_key="2e75bfc1f47b4b1f967f2e77e0e0119d",
    api_version="2024-02-15-preview",
    azure_endpoint="https://zif-chatbot.openai.azure.com/"
)

# Function to get Azure GPT response
def get_response(query):
    model = ChatVertexAIComponent()

    response = model.build(query, max_tokens=8000, temperature=0.5)

    return response

# Function to get text embeddings
def get_embedding(text):
    response = client.embeddings.create(input=[text], model='text-embedding-ada-002')
    return response.data[0].embedding

# Function to create embeddings for chunks
def create_embedding(chunks):
    embedding_list = []
    for chunk in chunks:
        chunk_embedding = get_embedding(chunk)
        embedding_list.append(chunk_embedding)
    return np.array(embedding_list)

# Function to split text into sentences without NLTK
def split_sentences(text):
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|\n)\s')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

# Function to chunk text into smaller pieces
def chunk_document(text, max_chunk_size=1000):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)
        if current_chunk_size + sentence_size > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_chunk_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

@app.post("/generate-questions")
async def generate_questions(file_link: str = Form(...), topic: str = Form(...)):
    # Step 1: Extract text from PDF
    text = ""
    try:
        reader = PdfReader(file_link)  # Reverting to file link/path as before
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += str(page.extract_text())
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to read PDF: {str(e)}"}, status_code=500)

    # Step 2: Chunk the large text
    chunks = chunk_document(text)

    # Step 3: Extract relevant chunks
    chunk_embeddings = create_embedding(chunks)
    topic_embedding = np.array(get_embedding(topic)).reshape(1, -1)

    # Calculate the cosine similarity between each chunk embedding and the topic embedding
    similarities = cosine_similarity(chunk_embeddings, topic_embedding)

    # Flatten the similarities array and get the top 2 most relevant chunks
    similarities = similarities.flatten()
    top_indices = np.argsort(similarities)[-2:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    relevant_content = str(relevant_chunks[0] + " " + relevant_chunks[1])

    # Step 4: Generate questions
    prompt = """
    <s>[INST]
    Generate 5 multiple-choice questions (MCQs) with answers and based on the following text without repeating the Questions:
    Text: '''{}'''
    Each MCQ should have four answer choices, and only one of them should be correct.
    Generate each question in the below JSON format, make sure that every question generated should be in the list.
    {{
      "question": "Question 1: [Your question here]",
      "choices": {{
        "A": "[Choice A]",
        "B": "[Choice B]",
        "C": "[Choice C]",
        "D": "[Choice D]"
      }},
      "answer": "[Correct choice, e.g., A]"
    }}
    [/INST]
    """.format(relevant_content)
    #print(prompt)
    response = get_response(prompt)

    
    lines = response.split("\n")
    response = (''.join(lines[1:-1]))
    #print(response)

    # Parse the response string into a JSON object
    try:
        mcqs = json.loads(response)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Unable to parse the response as valid JSON."}, status_code=500)

    return mcqs

if __name__ == "__main__":
    # print(get_response("How to Train a Neural Network in low end Hardware?"))
    # exit()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
