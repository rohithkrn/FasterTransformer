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
import torch
from transformers import AutoConfig

from .gptmodel import GPTModel
from .examples.gpt import parallel_gpt
from .utils.common_utils import verify_and_convert
import logging


class OPTModel(GPTModel):

    def create_ft_model_artifacts(self):
        cmd = "CUDA_VISIBLE_DEVICES=-1 "
        cmd += f"python {os.path.dirname(os.path.realpath(__file__))}/examples/gpt/huggingface_opt_convert.py " \
               f"-i {self.model} -o {self.DEFAULT_SAVE_DIR}/ -p {self.num_convert_process} " \
               f"-i_g {self.num_gpus} -weight_data_type {self.weight_dtype}"
        file_string = [os.path.join(self.DEFAULT_SAVE_DIR, f'{self.num_gpus}-gpu/verify'), self.verify_str]
        verify_and_convert(cmd, self.rank, file_string)

    def initialize(self):
        self.model_config = AutoConfig.from_pretrained(self.model)
        self.end_id = vars(self.model_config)['eos_token_id']
        logging.info("Start model artifacts conversion...")
        self.create_ft_model_artifacts()
        logging.info("load model...")
        self.load_gpt(os.path.join(self.DEFAULT_SAVE_DIR, f"{self.num_gpus}-gpu"), self.tensor_parallel_degree,
                      self.pipeline_parallel_degree, False, self.dtype, self.dtype)

    # TODO: support batch tokens
    def generate(self, start_ids: torch.Tensor, start_lengths: torch.IntTensor, batch_size, beam_width=1,
                 output_len=32, **kwargs):
        default_args = dict(
            beam_width=beam_width,
            top_k=1,
            top_p=0.0,
            temperature=1,
            repetition_penalty=1,
            random_seed=0,
        )
        default_args.update(kwargs)
        default_args["top_k"] *= torch.ones(batch_size, dtype=torch.int32)
        default_args["top_p"] *= torch.ones(batch_size, dtype=torch.float32)
        default_args["temperature"] *= torch.ones(batch_size, dtype=torch.float32)
        default_args["repetition_penalty"] *= torch.ones(batch_size, dtype=torch.float32)
        default_args["random_seed"] *= torch.ones(batch_size, dtype=torch.int64)
        with torch.no_grad():
            output = self.gpt(start_ids, start_lengths, output_len, **default_args)
        return output

    def load_gpt(self, model_path: str, tp: int, pp: int, use_int8: bool, inf_dtype, weight_dtype):
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
