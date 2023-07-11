# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
import os
import platform
import re

import torch
from dotenv import find_dotenv, load_dotenv


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    )

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
