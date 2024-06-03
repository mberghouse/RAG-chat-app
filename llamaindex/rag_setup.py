from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    ServiceContext, 
    load_index_from_storage
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
#from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
import os
from llama_index.llms.ollama import Ollama

from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
reader = SimpleDirectoryReader(input_dir="C:/Users/marcb/Downloads/trackpy/trackpy/")
documents = reader.load_data()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

splitter = SemanticSplitterNodeParser(
    buffer_size=1, 
    breakpoint_percentile_threshold=95, 
    embed_model=embed_model
)
nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

#llm = Groq(model="Llama3-70b-8192", api_key=GROQ_API_KEY)
llm = Ollama(model="llama3:8b", request_timeout=800.0)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, 
               service_context=service_context, node_parser=nodes)
vector_index.storage_context.persist(persist_dir="./storage")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, service_context=service_context)
cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=2)
query_engine = index.as_query_engine(service_context=service_context,
                similarity_top_k=10,
                node_postprocessors=[cohere_rerank],)
                