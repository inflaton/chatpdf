# -*- coding:utf-8 -*-
from __future__ import annotations

import csv
import datetime
import gc
import hashlib
import html
import json
import logging
import os
import platform
import re
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Type

import gradio as gr
import markdown2
import mdtex2html
import requests
import tiktoken
import torch
import transformers
from dotenv import find_dotenv, load_dotenv
from markdown import markdown
from peft import PeftModel
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import ClassNotFound, get_lexer_by_name, guess_lexer
from pypinyin import lazy_pinyin
from transformers import AutoModelForSeq2SeqLM, GenerationConfig, T5Tokenizer

from app_modules.presets import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


def markdown_to_html_with_syntax_highlight(md_str):
    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2)
        lang = lang.strip()
        # print(1,lang)
        if lang == "text":
            lexer = guess_lexer(code)
            lang = lexer.name
            # print(2,lang)
        try:
            lexer = get_lexer_by_name(lang, stripall=True)
        except ValueError:
            lexer = get_lexer_by_name("python", stripall=True)
        formatter = HtmlFormatter()
        # print(3,lexer.name)
        highlighted_code = highlight(code, lexer, formatter)

        return f'<pre><code class="{lang}">{highlighted_code}</code></pre>'

    code_block_pattern = r"```(\w+)?\n([\s\S]+?)\n```"
    md_str = re.sub(code_block_pattern, replacer, md_str, flags=re.MULTILINE)

    html_str = markdown(md_str)
    return html_str


def normalize_markdown(md_text: str) -> str:
    lines = md_text.split("\n")
    normalized_lines = []
    inside_list = False

    for i, line in enumerate(lines):
        if re.match(r"^(\d+\.|-|\*|\+)\s", line.strip()):
            if not inside_list and i > 0 and lines[i - 1].strip() != "":
                normalized_lines.append("")
            inside_list = True
            normalized_lines.append(line)
        elif inside_list and line.strip() == "":
            if i < len(lines) - 1 and not re.match(
                r"^(\d+\.|-|\*|\+)\s", lines[i + 1].strip()
            ):
                normalized_lines.append(line)
            continue
        else:
            inside_list = False
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def convert_mdtext(md_text):
    code_block_pattern = re.compile(r"```(.*?)(?:```|$)", re.DOTALL)
    inline_code_pattern = re.compile(r"`(.*?)`", re.DOTALL)
    code_blocks = code_block_pattern.findall(md_text)
    non_code_parts = code_block_pattern.split(md_text)[::2]

    result = []
    for non_code, code in zip(non_code_parts, code_blocks + [""]):
        if non_code.strip():
            non_code = normalize_markdown(non_code)
            if inline_code_pattern.search(non_code):
                result.append(markdown(non_code, extensions=["tables"]))
            else:
                result.append(mdtex2html.convert(non_code, extensions=["tables"]))
        if code.strip():
            code = f"\n```{code}\n\n```"
            code = markdown_to_html_with_syntax_highlight(code)
            result.append(code)
    result = "".join(result)
    result += ALREADY_CONVERTED_MARK
    return result


def convert_asis(userinput):
    return (
        f'<p style="white-space:pre-wrap;">{html.escape(userinput)}</p>'
        + ALREADY_CONVERTED_MARK
    )


def detect_converted_mark(userinput):
    if userinput.endswith(ALREADY_CONVERTED_MARK):
        return True
    else:
        return False


def detect_language(code):
    if code.startswith("\n"):
        first_line = ""
    else:
        first_line = code.strip().split("\n", 1)[0]
    language = first_line.lower() if first_line else ""
    code_without_language = code[len(first_line) :].lstrip() if first_line else code
    return language, code_without_language


def convert_to_markdown(text):
    text = text.replace("$", "&#36;")

    def replace_leading_tabs_and_spaces(line):
        new_line = []

        for char in line:
            if char == "\t":
                new_line.append("&#9;")
            elif char == " ":
                new_line.append("&nbsp;")
            else:
                break
        return "".join(new_line) + line[len(new_line) :]

    markdown_text = ""
    lines = text.split("\n")
    in_code_block = False

    for line in lines:
        if in_code_block is False and line.startswith("```"):
            in_code_block = True
            markdown_text += f"{line}\n"
        elif in_code_block is True and line.startswith("```"):
            in_code_block = False
            markdown_text += f"{line}\n"
        elif in_code_block:
            markdown_text += f"{line}\n"
        else:
            line = replace_leading_tabs_and_spaces(line)
            line = re.sub(r"^(#)", r"\\\1", line)
            markdown_text += f"{line}  \n"

    return markdown_text


def add_language_tag(text):
    def detect_language(code_block):
        try:
            lexer = guess_lexer(code_block)
            return lexer.name.lower()
        except ClassNotFound:
            return ""

    code_block_pattern = re.compile(r"(```)(\w*\n[^`]+```)", re.MULTILINE)

    def replacement(match):
        code_block = match.group(2)
        if match.group(2).startswith("\n"):
            language = detect_language(code_block)
            if language:
                return f"```{language}{code_block}```"
            else:
                return f"```\n{code_block}```"
        else:
            return match.group(1) + code_block + "```"

    text2 = code_block_pattern.sub(replacement, text)
    return text2


def delete_last_conversation(chatbot, history):
    if len(chatbot) > 0:
        chatbot.pop()

    if len(history) > 0:
        history.pop()

    return (
        chatbot,
        history,
        "Delete Done",
    )


def reset_state():
    return [], [], "Reset Done"


def reset_textbox():
    return gr.update(value=""), ""


def cancel_outputing():
    return "Stop Done"


def transfer_input(inputs):
    # 一次性返回，降低延迟
    textbox = reset_textbox()
    return (
        inputs,
        gr.update(value=""),
        gr.Button.update(visible=True),
    )


class State:
    interrupted = False

    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False


shared_state = State()


# Greedy Search
def greedy_search(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 25,
) -> Iterator[str]:
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # apply temperature
            logits /= temperature

            probs = torch.softmax(logits, dim=-1)
            # apply top_p
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0

            # apply top_k
            # if top_k is not None:
            #    probs_sort1, _ = torch.topk(probs_sort, top_k)
            #    min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
            #    probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)

            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)

            input_ids = torch.cat((input_ids, next_token), dim=-1)

            generated_tokens.append(next_token[0].item())
            text = tokenizer.decode(generated_tokens)

            yield text
            if any([x in text for x in stop_words]):
                del past_key_values
                del logits
                del probs
                del probs_sort
                del probs_idx
                del probs_sum
                gc.collect()
                return


def generate_prompt_with_history(text, history, tokenizer, max_length=2048):
    prompt = "The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n[|Human|]Hello!\n[|AI|]Hi!"
    history = ["\n[|Human|]{}\n[|AI|]{}".format(x[0], x[1]) for x in history]
    history.append("\n[|Human|]{}\n[|AI|]".format(text))
    history_text = ""
    flag = False
    for x in history[::-1]:
        if (
            tokenizer(prompt + history_text + x, return_tensors="pt")["input_ids"].size(
                -1
            )
            <= max_length
        ):
            history_text = x + history_text
            flag = True
        else:
            break
    if flag:
        return prompt + history_text, tokenizer(
            prompt + history_text, return_tensors="pt"
        )
    else:
        return None


def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False


def load_tokenizer_and_model(base_model, adapter_model=None, load_8bit=False):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model,
                device_map={"": device},
            )

    print(f"Model memory footprint: {model.get_memory_footprint()}")

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.eval()
    return tokenizer, model, device


class LogRecord(logging.LogRecord):
    def getMessage(self):
        msg = self.msg
        if self.args:
            if isinstance(self.args, dict):
                msg = msg.format(**self.args)
            else:
                msg = msg.format(*self.args)
        return msg


class Logger(logging.Logger):
    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        rv = LogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                rv.__dict__[key] = extra[key]
        return rv


def init_settings():
    logging.setLoggerClass(Logger)
    found_dotenv = find_dotenv(".env")
    if len(found_dotenv) == 0:
        found_dotenv = find_dotenv(".env.example")
    print(f"loading env vars from: {found_dotenv}")
    load_dotenv(found_dotenv, override=False)
    # print(f"loaded env vars: {os.environ}")


def remove_extra_spaces(text):
    return re.sub(" +", " ", text.strip())


def print_llm_response(llm_response):
    answer = llm_response["answer"] if "answer" in llm_response else None
    if answer is None:
        answer = llm_response["token"] if "token" in llm_response else None

    if answer is not None:
        print("\n\n***Answer:")
        print(remove_extra_spaces(answer))

    source_documents = (
        llm_response["source_documents"] if "source_documents" in llm_response else None
    )
    if source_documents is None:
        source_documents = llm_response["sourceDocs"]

    print("\nSources:")
    for source in source_documents:
        metadata = source["metadata"] if "metadata" in source else source.metadata
        print(
            "  Page: "
            + str(metadata["page"])
            + " Source: "
            + str(metadata["url"] if "url" in metadata else metadata["source"])
        )


def get_device_types():
    print("Running on: ", platform.platform())
    print("MPS is", "NOT" if not torch.backends.mps.is_available() else "", "available")
    print("CUDA is", "NOT" if not torch.cuda.is_available() else "", "available")
    device_type_available = "cpu"

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
    else:
        device_type_available = "mps"

    if torch.cuda.is_available():
        print("CUDA is available, we have found ", torch.cuda.device_count(), " GPU(s)")
        print(torch.cuda.get_device_name(0))
        print("CUDA version: " + torch.version.cuda)
        device_type_available = f"cuda:{torch.cuda.current_device()}"

    return (
        os.environ.get("HF_EMBEDDINGS_DEVICE_TYPE") or device_type_available,
        os.environ.get("HF_PIPELINE_DEVICE_TYPE") or device_type_available,
    )


if __name__ == "__main__":
    hf_embeddings_device_type, hf_pipeline_device_type = get_device_types()
    print(f"hf_embeddings_device_type: {hf_embeddings_device_type}")
    print(f"hf_pipeline_device_type: {hf_pipeline_device_type}")
