# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import configparser
import logging
import os
import math
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from .ftmodel import InferenceModel
from .examples.gpt.parallel_gpt import ParallelGPT
from .utils.common_utils import verify_and_convert


class GPTModel(InferenceModel):

    def __init__(self, model: str,
                 tensor_parallel_degree: int,
                 pipeline_parallel_degree: int, dtype: str, **kwargs):
        super().__init__(model, tensor_parallel_degree, pipeline_parallel_degree, dtype, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.gpt = None

    def initialize(self):
        logging.info("Converting hf model to ft model...")
        self.create_ft_model_artifacts()
        logging.info("Converting completed, start loading...")
        ckpt_config = configparser.ConfigParser()
        ckpt_config_path = os.path.join(self.DEFAULT_SAVE_DIR, f'{self.num_gpus}-gpu', 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
        self.start_id = ckpt_config.getint('gpt', 'start_id')
        self.end_id = ckpt_config.getint('gpt', 'end_id')
        layer_num = ckpt_config.getint('gpt', 'num_layer')
        head_num = ckpt_config.getint('gpt', 'head_num')
        size_per_head = ckpt_config.getint('gpt', 'size_per_head')
        vocab_size = ckpt_config.getint('gpt', 'vocab_size')
        max_seq_len = 1024
        shared_contexts_ratio = 1
        inference_data_type = self.dtype
        weights_data_type = self.dtype
        int8_mode = 0  # add 1 case
        gpt_with_moe = False
        expert_num = 0
        moe_layer_index = []
        moe_k = 0

        target_lib = os.path.join(self.lib_path, "libth_transformer.so")
        self.gpt = ParallelGPT(head_num, size_per_head, vocab_size, self.start_id, self.end_id,
                               layer_num, max_seq_len, self.tensor_parallel_degree, self.pipeline_parallel_degree,
                               lib_path=target_lib, inference_data_type=inference_data_type,
                               int8_mode=int8_mode, weights_data_type=weights_data_type,
                               shared_contexts_ratio=shared_contexts_ratio,
                               gpt_with_moe=gpt_with_moe,
                               expert_num=expert_num,
                               moe_k=moe_k,
                               moe_layer_index=moe_layer_index)
        if not self.gpt.load(ckpt_path=os.path.join(self.DEFAULT_SAVE_DIR, f'{self.num_gpus}-gpu')):
            raise IOError("Checkpoint file not found.")

    def generate(self, start_ids: torch.Tensor, start_lengths: torch.IntTensor, batch_size, beam_width=1,
                 output_len=32, **kwargs):
        default_args = dict(
            beam_width=beam_width,
            top_k=1,
            top_p=0.0,
            temperature=1,
            repetition_penalty=1.,
            random_seed=0,
            len_penalty=0,
            min_length=0,
            return_output_length=0,
            return_cum_log_probs=0
        )
        default_args.update(kwargs)
        default_args["top_k"] *= torch.ones(batch_size, dtype=torch.int32)
        default_args["top_p"] *= torch.ones(batch_size, dtype=torch.float32)
        default_args["temperature"] *= torch.ones(batch_size, dtype=torch.float32)
        default_args["repetition_penalty"] *= torch.ones(batch_size, dtype=torch.float32)
        default_args["random_seed"] *= torch.ones(batch_size, dtype=torch.int64)
        default_args["len_penalty"] *= torch.ones(batch_size, dtype=torch.float32)
        default_args["min_length"] *= torch.ones(batch_size, dtype=torch.int32)

        with torch.no_grad():
            result = self.gpt(start_ids, start_lengths, output_len, **default_args)
        return result

    def pipeline_generate(self, inputs, batch_size=1, output_len=32, beam_width=1,
                          skip_end_tokens=True, detokenize=True, **kwargs):
        total_iter = math.ceil(len(inputs) / batch_size)
        result = []
        for it in range(total_iter):
            input_batch = inputs[it * batch_size: batch_size * (it + 1)]
            start_ids = [torch.tensor(self.tokenizer.encode(input), dtype=torch.int32, device=self.device) for input in
                         input_batch]
            start_lengths = [len(ids) for ids in start_ids]
            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=self.end_id)
            start_lengths = torch.IntTensor(start_lengths)
            tokens_batch = self.generate(start_ids, start_lengths, batch_size, beam_width, output_len, **kwargs)

            outputs = []
            tokens_batch = tokens_batch.cpu().numpy()
            for i, (input, tokens) in enumerate(zip(inputs, tokens_batch)):
                for beam_id in range(beam_width):
                    token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                    if skip_end_tokens:
                        token = token[token != self.end_id]
                    output = self.tokenizer.decode(token) if detokenize else ' '.join(str(t) for t in token.tolist())
                    outputs.append(output)

            result.append(outputs)
        return result

    def create_ft_model_artifacts(self):
        cmd = "CUDA_VISIBLE_DEVICES=-1 "
        cmd += f"python {os.path.dirname(os.path.realpath(__file__))}/examples/gpt/huggingface_gpt_convert.py " \
               f"-i {self.model} -o {self.DEFAULT_SAVE_DIR}/ -p {self.num_convert_process} " \
               f"-i_g {self.num_gpus} -weight_data_type {self.weight_dtype}"
        file_string = [os.path.join(self.DEFAULT_SAVE_DIR, f'{self.num_gpus}-gpu/verify'), self.verify_str]
        verify_and_convert(cmd, self.rank, file_string)
