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

from app_modules.presets import *
from app_modules.qa_chain import QAChain
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
chat_history_enabled = os.environ.get("CHAT_HISTORY_ENABLED") == "true"
show_param_settings = os.environ.get("SHOW_PARAM_SETTINGS") == "true"
share_gradio_app = os.environ.get("SHARE_GRADIO_APP") == "true"


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


def qa(chatbot):
    user_msg = chatbot[-1][0]
    q = Queue()
    result = Queue()
    job_done = object()

    def task(question, chat_history):
        start = timer()
        ret = qa_chain.call({"question": question, "chat_history": chat_history}, q)
        end = timer()

        print(f"Completed in {end - start:.3f}s")
        print_llm_response(ret)

        q.put(job_done)
        result.put(ret)

    with start_blocking_portal() as portal:
        chat_history = []
        if chat_history_enabled:
            for i in range(len(chatbot) - 1):
                element = chatbot[i]
                item = (element[0] or "", element[1] or "")
                chat_history.append(item)

        portal.start_task_soon(task, user_msg, chat_history)

        content = ""
        count = 2 if len(chat_history) > 0 else 1

        while count > 0:
            while q.empty():
                print("nothing generated yet - retry in 0.5s")
                time.sleep(0.5)

            for next_token in qa_chain.streamer:
                if next_token is job_done:
                    break
                content += next_token or ""
                chatbot[-1][1] = remove_extra_spaces(content)

                if count == 1:
                    yield chatbot

            count -= 1

        chatbot[-1][1] += "\n\nSources:\n"
        ret = result.get()
        titles = []
        for doc in ret["source_documents"]:
            page = doc.metadata["page"] + 1
            url = f"{doc.metadata['url']}#page={page}"
            file_name = doc.metadata["source"].split("/")[-1]
            title = f"{file_name} Page: {page}"
            if title not in titles:
                titles.append(title)
                chatbot[-1][1] += f"1. [{title}]({url})\n"

        yield chatbot


with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    user_question = gr.State("")
    with gr.Row():
        gr.HTML(title)
    gr.Markdown(description_top)
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row():
                chatbot = gr.Chatbot(elem_id="inflaton_chatbot").style(height="100%")
            with gr.Row():
                with gr.Column(scale=2):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Enter your question here"
                    ).style(container=False)
                with gr.Column(
                    min_width=70,
                ):
                    submitBtn = gr.Button("Send")
                with gr.Column(
                    min_width=70,
                ):
                    clearBtn = gr.Button("Clear")
        if show_param_settings:
            with gr.Column():
                with gr.Column(
                    min_width=50,
                ):
                    with gr.Tab(label="Parameter Setting"):
                        gr.Markdown("# Parameters")
                        top_p = gr.Slider(
                            minimum=-0,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            # interactive=True,
                            label="Top-p",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0,
                            step=0.1,
                            # interactive=True,
                            label="Temperature",
                        )
                        max_new_tokens = gr.Slider(
                            minimum=0,
                            maximum=2048,
                            value=2048,
                            step=8,
                            # interactive=True,
                            label="Max Generation Tokens",
                        )
                        max_context_length_tokens = gr.Slider(
                            minimum=0,
                            maximum=4096,
                            value=4096,
                            step=128,
                            # interactive=True,
                            label="Max Context Tokens",
                        )
    gr.Markdown(description)

    def chat(user_message, history):
        return "", history + [[user_message, None]]

    user_input.submit(
        chat, [user_input, chatbot], [user_input, chatbot], queue=True
    ).then(qa, chatbot, chatbot)

    submitBtn.click(
        chat, [user_input, chatbot], [user_input, chatbot], queue=True
    ).then(qa, chatbot, chatbot)

    def reset():
        return "", []

    clearBtn.click(
        reset,
        outputs=[user_input, chatbot],
        show_progress=True,
    )

demo.title = "Chat with PCI DSS v4"
demo.queue(concurrency_count=1, api_open=False).launch(share=share_gradio_app)
