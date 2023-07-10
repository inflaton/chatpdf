import os
import sys
from timeit import default_timer as timer
from typing import List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import GPT4All
from langchain.schema import LLMResult
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS

from app_modules.qa_chain import *
from app_modules.utils import *

# Constants
init_settings()

# https://github.com/huggingface/transformers/issues/17611
os.environ["CURL_CA_BUNDLE"] = ""

hf_embeddings_device_type, hf_pipeline_device_type = get_device_types()
print(f"hf_embeddings_device_type: {hf_embeddings_device_type}")
print(f"hf_pipeline_device_type: {hf_pipeline_device_type}")

hf_embeddings_model_name = (
    os.environ.get("HF_EMBEDDINGS_MODEL_NAME") or "hkunlp/instructor-xl"
)
n_threds = int(os.environ.get("NUMBER_OF_CPU_CORES") or "4")
index_path = os.environ.get("FAISS_INDEX_PATH") or os.environ.get("CHROMADB_INDEX_PATH")
using_faiss = os.environ.get("FAISS_INDEX_PATH") is not None
llm_model_type = os.environ.get("LLM_MODEL_TYPE")
chatting = len(sys.argv) > 1 and sys.argv[1] == "chat"
questions_file_path = os.environ.get("QUESTIONS_FILE_PATH")
chat_history_enabled = os.environ.get("CHAT_HISTORY_ENABLED") or "true"

## utility functions

import os


class MyCustomHandler(BaseCallbackHandler):
    def __init__(self):
        self.reset()

    def reset(self):
        self.texts = []

    def get_standalone_question(self) -> str:
        return self.texts[0].strip() if len(self.texts) > 0 else None

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when chain ends running."""
        print("\non_llm_end - response:")
        print(response)
        self.texts.append(response.generations[0][0].text)


start = timer()
embeddings = HuggingFaceInstructEmbeddings(
    model_name=hf_embeddings_model_name,
    model_kwargs={"device": hf_embeddings_device_type},
)
end = timer()

print(f"Completed in {end - start:.3f}s")

start = timer()

print(f"Load index from {index_path} with {'FAISS' if using_faiss else 'Chroma'}")

if not os.path.isdir(index_path):
    raise ValueError(f"{index_path} does not exist!")
elif using_faiss:
    vectorstore = FAISS.load_local(index_path, embeddings)
else:
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=index_path)

end = timer()

print(f"Completed in {end - start:.3f}s")

start = timer()
qa_chain = QAChain(vectorstore, llm_model_type)
custom_handler = MyCustomHandler()
qa_chain.init(
    custom_handler, n_threds=n_threds, hf_pipeline_device_type=hf_pipeline_device_type
)
qa = qa_chain.get_chain()
end = timer()
print(f"Completed in {end - start:.3f}s")

# input("Press Enter to continue...")
# exit()

# Chatbot loop
chat_history = []
print("Welcome to the ChatPDF! Type 'exit' to stop.")

# Open the file for reading
file = open(questions_file_path, "r")

# Read the contents of the file into a list of strings
queue = file.readlines()
for i in range(len(queue)):
    queue[i] = queue[i].strip()

# Close the file
file.close()

queue.append("exit")

chat_start = timer()

while True:
    if chatting:
        query = input("Please enter your question: ")
    else:
        query = queue.pop(0)

    query = query.strip()
    if query.lower() == "exit":
        break

    print("\nQuestion: " + query)
    custom_handler.reset()

    start = timer()
    result = qa({"question": query, "chat_history": chat_history})
    end = timer()
    print(f"Completed in {end - start:.3f}s")

    print_llm_response(result)

    if len(chat_history) == 0:
        standalone_question = query
    else:
        standalone_question = custom_handler.get_standalone_question()

    if standalone_question is not None:
        print(f"Load relevant documents for standalone question: {standalone_question}")
        start = timer()
        docs = qa.retriever.get_relevant_documents(standalone_question)
        end = timer()

        print(docs)
        print(f"Completed in {end - start:.3f}s")

    if chat_history_enabled == "true":
        chat_history.append((query, result["answer"]))

chat_end = timer()
print(f"Total time used: {chat_end - chat_start:.3f}s")
