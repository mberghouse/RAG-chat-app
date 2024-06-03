import os
import json
from dotenv import load_dotenv
from llama_index.core import StorageContext, ServiceContext, load_index_from_storage
from llama_index.core.callbacks.base import CallbackManager
#from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
import chainlit as cl
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CONVERSATION_HISTORY_FILE = "./storage/conversation_history.json"

def save_conversation_history(history):
    try:
        with open(CONVERSATION_HISTORY_FILE, "w") as f:
            json.dump(history, f)
        print("Conversation history saved successfully.")
    except Exception as e:
        print(f"Failed to save conversation history: {e}")

def load_conversation_history():
    try:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            with open(CONVERSATION_HISTORY_FILE, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Failed to load conversation history: {e}")
        return []

def clear_conversation_history():
    try:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            os.remove(CONVERSATION_HISTORY_FILE)
            print("Conversation history file removed successfully.")
        else:
            print("No conversation history file to remove.")
    except Exception as e:
        print(f"Failed to remove conversation history file: {e}")

    # Clear in-memory session history
    cl.user_session.set("conversation_history", [])

@cl.on_chat_start
async def factory():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    #llm = Groq(model="Llama3-70b-8192", api_key=GROQ_API_KEY)
    llm = Ollama(model="llama3:8b", request_timeout=800.0)
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=llm,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])
    )
    cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=2)

    index = load_index_from_storage(storage_context, service_context=service_context)

    query_engine = index.as_query_engine(
        service_context=service_context,
        similarity_top_k=10,
        node_postprocessors=[cohere_rerank],
        # streaming=True,
    )

    cl.user_session.set("query_engine", query_engine)

    # Load conversation history from file
    conversation_history = load_conversation_history()
    cl.user_session.set("conversation_history", conversation_history)
    print("Conversation history loaded successfully.")
    print(conversation_history)

@cl.on_message
async def main(message: cl.Message):
    if message.content.strip().lower() == "/clear":
        clear_conversation_history()
        await cl.Message(content="Conversation history cleared.").send()
        return

    query_engine = cl.user_session.get("query_engine")
    conversation_history = cl.user_session.get("conversation_history")

    # Combine the history into the query context if necessary
    history_text = "\n".join(f"Q: {interaction['prompt']}\nA: {interaction['response']}" for interaction in conversation_history)
    query_text = f"{history_text}\nQ: {message.content}"

    response = await cl.make_async(query_engine.query)(query_text)

    response_message = cl.Message(content="")

    for token in response.response:
        await response_message.stream_token(token=token)

    await response_message.send()

    # Save the conversation to history
    conversation_history.append({"prompt": message.content, "response": response.response})
    cl.user_session.set("conversation_history", conversation_history)

    # Save conversation history to file
    save_conversation_history(conversation_history)

    # Print conversation history for debugging
    print("Updated Conversation History:")
    for interaction in conversation_history:
        print("Prompt:", interaction["prompt"])
        print("Response:", interaction["response"])

# Testing clear function
if __name__ == "__main__":
    clear_conversation_history()


