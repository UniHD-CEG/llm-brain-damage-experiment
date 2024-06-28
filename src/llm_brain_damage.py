#!/usr/bin/env python3

# Copyright (C) 2024 Franz Kevin Stehle Computing Systems Group, Institute of Computer Engineering, Heidelberg University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import random
from contextlib import contextmanager
from typing import Callable, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.quantizers import HfQuantizer
import gradio as gr

sys.path.append('.')

from bitflipnoise import BitFlipNoise

MACHINE_TYPE = 'cuda'

# Add your desired server URL/IP and port here

GRADIO_SERVER_URL = ''
GRADIO_SERVER_PORT = 7680

start_prompt = '<start_of_turn>user\nYour name is Gemma, a light-weight and open source LLM released by Google.'\
                                                        '<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n'


tokenizer = AutoTokenizer.from_pretrained("../models/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("../models/gemma-7b-it",
                                                    torch_dtype=torch.bfloat16,
                                                    device_map='sequential',
                                                    attn_implementation="flash_attention_2")

noise_level = 0

bit_flip_noise = BitFlipNoise((0, 0))


def add_rram_write_noise(layer: nn.Module, inputs: Tuple[torch.Tensor], _output: torch.Tensor):

    if noise_level > 0:
        if isinstance(_output, Tuple):

            _output_noisy = []

            for element in _output:
                if isinstance(element, torch.Tensor):
                    bit_flip_noise(element)
                _output_noisy.append(element)

            return tuple(_output_noisy)

        elif isinstance(_output, torch.Tensor):
            bit_flip_noise(_output)
            return _output


def separate_queries_and_answers(history: str):
    '''
    Returns a tuple containing the queries
    as first and the answers as the second element
    '''
    if len(history):
        return map(list, zip(*history))
    else:
        return [], []

def format_prompt(message, history):

    prompt = ''

    if history:
        queries, answers = separate_queries_and_answers(history)

        for user_prompt, bot_response in zip(queries, answers):
            prompt += f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            prompt += f"<start_of_turn>model\n{bot_response.strip().removesuffix('<eos>')}<end_of_turn>\n"

    else:
        prompt += start_prompt

    prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model"

    return prompt


def chat_func(message, history, noise_level_in):

    global noise_level
    global bit_flip_noise

    if noise_level != noise_level_in:
        noise_level = noise_level_in

        bit_flip_noise = BitFlipNoise((noise_level,
                                            noise_level))

    prompt = format_prompt(message, history)

    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Add the RRAM write noise hook to every module in the model

    handle = nn.modules.module.register_module_forward_hook(add_rram_write_noise)

    outputs = model.generate(**input_ids,
                                do_sample=True,
                                temperature=0.4,
                                top_k=5,
                                top_p=0.9,
                                max_new_tokens=2048)

    response = tokenizer.decode(outputs[0])

    handle.remove()
    torch.cuda.empty_cache()

    response = response.rsplit('<start_of_turn>model', maxsplit=1)[1]

    return response

read_noise = gr.Slider(label='Activation Write Noise',
                        value=0, minimum=0, maximum=1e-4, step=1e-8,
                        interactive=True, visible=True)

demo = gr.ChatInterface(chat_func,
                            title='LLM Brain Damage Demo',
                            description='Exploration of the Effects of RRAM Write Noise in bfloat16 Activations',
                            additional_inputs=read_noise)

demo.queue().launch(server_name=GRADIO_SERVER_URL,
                        server_port=GRADIO_SERVER_PORT)

