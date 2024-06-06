import os
import csv
import time
import streamlit as st
from rouge import Rouge
from llama_index.core import SimpleDirectoryReader
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.legacy.core.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.legacy.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.legacy import ServiceContext
from llama_index.legacy.llms import OpenAI
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.legacy.retrievers import VectorIndexRetriever
from llama_index.legacy.retrievers import BaseRetriever
from llama_index.legacy.chat_engine import CondensePlusContextChatEngine
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.postprocessor import LongContextReorder
from llama_index.legacy.schema import MetadataMode
from llama_index.legacy.schema import QueryBundle
from llama_index.legacy import (StorageContext, load_index_from_storage)
import openai
import nest_asyncio

# Set custom NLTK data directory
nltk_data_dir = '/home/adminuser/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_dir

nest_asyncio.apply()

# openai.api_key = os.environ.get("SECRET_TOKEN")
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

st.set_page_config(page_title="Chat with POM Course Material, powered by AIXplorers", page_icon="✅", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.environ['SECRET_TOKEN']
st.title("Chat with your Course, developed by [GurukulAI](https://www.linkedin.com/in/sai-ganesh-91505a261?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)!! 💬")

with st.sidebar:
    st.title('🤗💬Chat with I-Venture @ ISB')
    st.success('Access to this Gen-AI Powered Chatbot is provided by [Anupam](https://anupam-purwar.github.io/page/research_group.html)!!', icon='✅')
    hf_email = 'anupam_purwar2019@pgp.isb.edu'
    hf_pass = 'PASS'

DEFAULT_CONTEXT_PROMPT_TEMPLATE_1 = """
 You're an AI assistant to help students learn their course material via conversations.
 The following is a friendly conversation between a user and an AI assistant for answering questions related to query.
 The assistant is talkative and provides lots of specific details in form of bullet points or short paras from the context.
 Here is the relevant context:


 {context_str}


 Instruction: Based on the above context, provide a detailed answer IN THE USER'S LANGUAGE with logical formation of paragraphs for the user question below.
 """

condense_prompt = (
  "Given the following conversation between a user and an AI assistant and a follow-up question from the user,"
  "rephrase the follow-up question to be a standalone question.\n"
  "Chat History:\n"
  "{chat_history}"
  "\nFollow-Up Input: {question}"
  "\nStandalone question:"
)

def indexgenerator(indexPath, documentsPath):
    if not os.path.exists(indexPath):
        print("Not existing")
        entity_extractor = EntityExtractor(prediction_threshold=0.2, label_entities=False, device="cpu") # set device to "cuda" if gpu exists
        node_parser = SentenceSplitter(chunk_overlap=100, chunk_size=1024)
        transformations = [node_parser, entity_extractor]
        documents = SimpleDirectoryReader(input_dir=documentsPath).load_data()
        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0), embed_model=embed_model)
        index = VectorStoreIndex(nodes, service_context=service_context)
        index.storage_context.persist(indexPath)
    else:
        print("Existing")
        storage_context = StorageContext.from_defaults(persist_dir=indexPath)
        index = load_index_from_storage(storage_context, service_context=ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0), embed_model=embed_model))
    return index

indexPath = '/IndicesClassified/dlabs-indices'
documentsPath = ''
index = indexgenerator(indexPath, documentsPath)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
topk = 4
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=topk)
postprocessor = LongContextReorder()
bm25_flag = True
try:
    bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=topk)
except:
    source_nodes = index.docstore.docs.values()
    nodes = list(source_nodes)
    bm25_flag = False

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        all_nodes = bm25_nodes + vector_nodes
        query = str(query)
        all_nodes = postprocessor.postprocess_nodes(nodes=all_nodes, query_bundle=QueryBundle(query_str=query.lower()))
        return all_nodes[0:topk]

if bm25_flag:
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
else:
    hybrid_retriever = vector_retriever

query_engine = RetrieverQueryEngine.from_args(retriever=hybrid_retriever, service_context=service_context, verbose=True)

def main():
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about I-Venture @ ISB"}
        ]
    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(query_engine, context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE)
    
    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": str(prompt)})
    
    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                all_nodes = hybrid_retriever.retrieve(str(prompt))
                start = time.time()
                response = st.session_state.chat_engine.chat(str(prompt))
                end = time.time()
                time_taken = end - start
                st.write(response.response)
                context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in all_nodes])
                scores = rouge.get_scores(response.response, context_str)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
                unigram_recall = scores[0]["rouge-1"]["r"]
                unigram_precision = scores[0]["rouge-1"]["p"]
                bigram_recall = scores[0]["rouge-2"]["r"]
                bigram_precision = scores[0]["rouge-2"]["p"]
                st.write(f"Time taken: {time_taken:.2f} seconds")
                st.write(f"Unigram Recall: {unigram_recall:.2f}")
                st.write(f"Unigram Precision: {unigram_precision:.2f}")
                st.write(f"Bigram Recall: {bigram_recall:.2f}")
                st.write(f"Bigram Precision: {bigram_precision:.2f}")

if __name__ == "__main__":
    main()
