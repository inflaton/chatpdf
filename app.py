"""Main entrypoint for the app."""
import os
import time
from queue import Queue
from timeit import default_timer as timer

import gradio as gr
from anyio.from_thread import start_blocking_portal
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS

from app_modules.qa_chain import QAChain
from app_modules.utils import get_device_types, init_settings, remove_extra_spaces

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
chat_history_enabled = os.environ.get("CHAT_HISTORY_ENABLED") or "true"

streaming_enabled = True  # llm_model_type in ["openai", "llamacpp"]

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
qa_chain.init(n_threds=n_threds, hf_pipeline_device_type=hf_pipeline_device_type)
end = timer()
print(f"Completed in {end - start:.3f}s")


def bot(chatbot):
    user_msg = chatbot[-1][0]

    prompt = user_msg
    q = Queue()
    job_done = object()

    def task(question):
        chat_history = []
        if chat_history_enabled == "true":
            for i in range(len(chatbot) - 1):
                element = chatbot[i]
                item = (element[0] or "", element[1] or "")
                chat_history.append(item)

        start = timer()
        ret = qa_chain.call({"question": question, "chat_history": chat_history}, q)
        end = timer()
        print(f"Completed in {end - start:.3f}s")
        q.put(job_done)
        print(f"sources:\n{ret['source_documents']}")
        return ret

    with start_blocking_portal() as portal:
        portal.start_task_soon(task, prompt)

        content = ""
        while True:
            try:
                next_token = q.get(True, timeout=1)
                if next_token is job_done:
                    break
                content += next_token or ""
                chatbot[-1][1] = remove_extra_spaces(content)

                yield chatbot
            except Exception:
                print("nothing generated yet - retry in 0.5s")
                time.sleep(0.5)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Question")

    def chat(user_message, history):
        return "", history + [[user_message, None]]

    msg.submit(chat, [msg, chatbot], [msg, chatbot], queue=True).then(
        bot, chatbot, chatbot
    )

demo.queue()
demo.launch()
