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

from .ftmodel import InferenceModel
from .examples.gpt import gpt_token_encoder as encoder
from .examples.gpt import comm
from .examples.gpt.parallel_gpt import ParallelGPT
from .utils.common_utils import execute_command


class GPTModel(InferenceModel):
    DEFAULT_SAVE_DIR = os.path.join(os.getcwd() + "/ft_gpt_model/")
    DEFAULT_INPUT_DIR = os.path.join(os.getcwd() + "/gpt2-xl")

    def __init__(self, model, tensor_parallel_degree, pipeline_parallel_degree, **kwargs):
        super().__init__(model, tensor_parallel_degree, pipeline_parallel_degree, **kwargs)
        logging.info("Converting hf model to ft model...")
        self.create_ft_model_artifacts()
        self.ckpt_config = configparser.ConfigParser()
        ckpt_config_path = kwargs.get("ckpt_config_path",
                                      os.path.join(self.DEFAULT_SAVE_DIR, f'{self.num_gpus}-gpu', 'config.ini'))
        if os.path.isfile(ckpt_config_path):
            self.ckpt_config.read(ckpt_config_path)
        self.infer_decode_args = None
        self.vocab_file = kwargs.get("vocab_file", os.path.join(self.DEFAULT_INPUT_DIR, "vocab.json"))
        self.merges_file = kwargs.get("merges_file", os.path.join(self.DEFAULT_INPUT_DIR, "merges.txt"))
        self.enc = encoder.get_encoder(self.vocab_file, self.merges_file)
        self.start_id = self.ckpt_config.getint('gpt', 'start_id')
        self.end_id = self.ckpt_config.getint('gpt', 'end_id')
        self.beam_width = kwargs.get("beam_width", 1)
        self.skip_end_tokens = kwargs.get("skip_end_tokens", True)
        self.detokenize = kwargs.get("detokenize", True)

    def initialize(self):
        ## TODO: ideall we should read these from a config file use defines. For now, default values are set for these args
        layer_num = self.ckpt_config.getint('gpt', 'num_layer')
        output_len = 32
        head_num = self.ckpt_config.getint('gpt', 'head_num')
        size_per_head = self.ckpt_config.getint('gpt', 'size_per_head')
        vocab_size = self.ckpt_config.getint('gpt', 'vocab_size')
        top_k = 1
        top_p = 0
        temperature = 1
        len_penalty = 0
        beam_search_diversity_rate = 0
        max_seq_len = 1024
        repetition_penalty = 1
        presence_penalty = 0
        min_length = 0
        weights_data_type = self.ckpt_config.get('gpt', 'weight_data_type')
        return_cum_log_probs = 0
        return_output_length = return_cum_log_probs > 0
        shared_contexts_ratio = 1
        inference_data_type = "fp32"  # add others
        int8_mode = 0  # add 1 case
        enable_random_seed = False
        gpt_with_moe = False
        expert_num = 0
        moe_layer_index = []
        moe_k = 0

        comm.initialize_model_parallel(self.tensor_parallel_degree, self.pipeline_parallel_degree)
        self.rank = comm.get_rank()
        self.device = comm.get_device()

        target_lib = os.path.join(self.lib_path, "libth_transformer.so")
        self.model = ParallelGPT(head_num, size_per_head, vocab_size, self.start_id, self.end_id,
                                 layer_num, max_seq_len, self.tensor_parallel_degree, self.pipeline_parallel_degree,
                                 lib_path=target_lib, inference_data_type=inference_data_type,
                                 int8_mode=int8_mode, weights_data_type=weights_data_type,
                                 shared_contexts_ratio=shared_contexts_ratio,
                                 gpt_with_moe=gpt_with_moe,
                                 expert_num=expert_num,
                                 moe_k=moe_k,
                                 moe_layer_index=moe_layer_index)
        if not self.model.load(ckpt_path=os.path.join(self.DEFAULT_SAVE_DIR, f'{self.num_gpus}-gpu')):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")

        if enable_random_seed:
            random_seed_tensor = torch.randint(0, 10000, size=[self.batch_size], dtype=torch.int64)
        else:
            random_seed_tensor = torch.zeros([self.batch_size], dtype=torch.int64)

        bad_words_list = None

        repetition_penalty_vec = None if repetition_penalty == 1. else repetition_penalty * torch.ones(self.batch_size,
                                                                                                       dtype=torch.float32)
        presence_penalty_vec = None if presence_penalty == 0. else presence_penalty * torch.ones(self.batch_size,
                                                                                                 dtype=torch.float32)

        self.infer_decode_args = dict(
            beam_width=self.beam_width,
            top_k=top_k * torch.ones(self.batch_size, dtype=torch.int32),
            top_p=top_p * torch.ones(self.batch_size, dtype=torch.float32),
            temperature=temperature * torch.ones(self.batch_size, dtype=torch.float32),
            repetition_penalty=repetition_penalty_vec,
            presence_penalty=presence_penalty_vec,
            beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(self.batch_size, dtype=torch.float32),
            len_penalty=len_penalty * torch.ones(size=[self.batch_size], dtype=torch.float32),
            bad_words_list=bad_words_list,
            min_length=min_length * torch.ones(size=[self.batch_size], dtype=torch.int32),
            random_seed=random_seed_tensor
        )

    def generate(self, inputs, output_len=32, return_output_length=0, return_cum_log_probs=0):
        total_iter = math.ceil(len(inputs) / self.batch_size)
        result = []
        for it in range(total_iter):
            input_batch = inputs[it * self.batch_size: self.batch_size * (it + 1)]
            start_ids = [torch.tensor(self.enc.encode(input), dtype=torch.int32, device=self.device) for input in
                         input_batch]
            start_lengths = [len(ids) for ids in start_ids]
            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=self.end_id)
            start_lengths = torch.IntTensor(start_lengths)
            tokens_batch = self.model(start_ids,
                                      start_lengths,
                                      output_len,
                                      return_output_length=return_output_length,
                                      return_cum_log_probs=return_cum_log_probs,
                                      **self.infer_decode_args)

            outputs = []
            tokens_batch = tokens_batch.cpu().numpy()
            for i, (input, tokens) in enumerate(zip(inputs, tokens_batch)):
                for beam_id in range(self.beam_width):
                    token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                    if self.skip_end_tokens:
                        token = token[token != self.end_id]
                    output = self.enc.decode(token) if self.detokenize else ' '.join(str(t) for t in token.tolist())
                    outputs.append(output)

            result.append(outputs)
        return result

    def create_ft_model_artifacts(self):
        cmd = f"python {os.path.dirname(os.path.realpath(__file__))}/examples/gpt/huggingface_gpt_convert.py " \
              f"-i {self.model} -o {self.DEFAULT_SAVE_DIR} -i_g {self.num_gpus}"
        execute_command(cmd, self.rank)
