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
import os
import numpy as np
import torch
from transformers import AutoConfig
from .examples.gpt import parallel_gpt
from .ftmodel import InferenceModel
from .utils.common_utils import execute_command
import logging


class OPTModel(InferenceModel):

    # TODO: optimize this
    DEFAULT_SAVING_LOC = "/opt/djl/ft_model/opt"

    def __init__(self, model: str, tensor_parallel_degree, pipeline_parallel_degree, dtype=np.float32,
                 batch_size=1, **kwargs):
        super().__init__(model, tensor_parallel_degree, pipeline_parallel_degree, dtype, batch_size, **kwargs)
        self.gpt: parallel_gpt.ParallelGPT = None
        self.model_config = AutoConfig.from_pretrained(self.model)

    def create_ft_model_artifacts(self):
        cmd = f"python {os.path.dirname(os.path.realpath(__file__))}/examples/gpt/huggingface_opt_convert.py " \
              f"-i {self.model} -o {self.DEFAULT_SAVING_LOC}/ -i_g {self.num_gpus}"
        execute_command(cmd, self.rank)

    def initialize(self):
        logging.info("Start model artifacts conversion...")
        self.create_ft_model_artifacts()
        logging.info("load model...")
        self.load_gpt(os.path.join(self.DEFAULT_SAVING_LOC, f"{self.num_gpus}-gpu"), self.tensor_parallel_degree,
                      self.pipeline_parallel_degree, False, 'fp16', 'fp16')

    # TODO: support batch tokens
    def generate(self, input_tokens: torch.IntTensor, output_length: int):
        top_k = 1
        top_p = 0.0
        max_batch_size = 1
        temperature = 1
        repetition_penalty = 1
        random_seed = 0
        input_lengths = torch.tensor([len(input_tokens)], dtype=torch.int32, device=self.gpt.device)
        infer_decode_args = dict(
            beam_width=1,
            top_k=top_k * torch.ones(max_batch_size, dtype=torch.int32),
            top_p=top_p * torch.ones(max_batch_size, dtype=torch.float32),
            temperature=temperature * torch.ones(max_batch_size, dtype=torch.float32),
            repetition_penalty=repetition_penalty * torch.ones(max_batch_size, dtype=torch.float32),
            random_seed=random_seed * torch.ones(max_batch_size, dtype=torch.int64)
        )
        with torch.no_grad():
            output, ft_output_len = self.gpt(input_tokens, input_lengths,
                                             output_length, **infer_decode_args)
        tokens = output[0][0]
        return tokens

    def load_gpt(self, model_path: str, tp: int, pp: int, use_int8: bool, inf_dtype,
                 weight_dtype):
        hf_config = vars(self.model_config)
        layernorm_eps = 1e-5
        layernorm_type = 'pre_layernorm' if hf_config['do_layer_norm_before'] else 'post_layernorm'
        has_post_decoder_layernorm = layernorm_type == 'pre_layernorm'
        activation_type = 'Relu' if hf_config['activation_function'] == 'relu' else 'Gelu'
        target_lib = os.path.join(self.lib_path, "libth_transformer.so")
        gpt = parallel_gpt.ParallelGPT(
            hf_config['num_attention_heads'],
            hf_config['hidden_size'] // hf_config['num_attention_heads'],
            hf_config['vocab_size'],
            hf_config['bos_token_id'],
            hf_config['eos_token_id'],
            hf_config['num_hidden_layers'],
            hf_config['max_position_embeddings'],
            tp, pp, target_lib,
            layernorm_eps=layernorm_eps,
            layernorm_type=layernorm_type,
            activation_type=activation_type,
            has_post_decoder_layernorm=has_post_decoder_layernorm,
            int8_mode=1 if use_int8 else 0,
            inference_data_type=inf_dtype,
            weights_data_type=weight_dtype)
        if not gpt.load(model_path):
            raise ValueError(f"model artifacts not found {model_path}")
        self.gpt = gpt
