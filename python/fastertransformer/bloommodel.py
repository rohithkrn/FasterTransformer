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

from transformers import AutoConfig, AutoTokenizer

from .gptmodel import GPTModel
from .examples.gpt import bloom
from .utils.common_utils import verify_and_convert


class BLOOMModel(GPTModel):

    def create_ft_model_artifacts(self):
        cmd = f"python {os.path.dirname(os.path.realpath(__file__))}/examples/gpt/huggingface_bloom_convert.py " \
              f"-i {self.model} -o {self.DEFAULT_SAVE_DIR}/ -p {self.num_convert_process} " \
              f"-tp {self.num_gpus} -dt {self.weight_dtype}"
        file_string = [os.path.join(self.DEFAULT_SAVE_DIR, f'{self.num_gpus}-gpu/verify'), self.verify_str]
        verify_and_convert(cmd, self.rank, file_string)

    def initialize(self):
        padding_side = 'right'  # FT exclusive
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, padding_side=padding_side)
        self.model_config = AutoConfig.from_pretrained(self.model)
        self.end_id = vars(self.model_config)['eos_token_id']
        logging.info("Start model artifacts conversion...")
        self.create_ft_model_artifacts()
        logging.info("load model...")
        self.gpt = self.get_model()

    def get_model(self):
        ckpt_path = os.path.join(self.DEFAULT_SAVE_DIR, f'{self.num_gpus}-gpu')
        config_path = os.path.join(ckpt_path, 'config.ini')
        if os.path.isfile(config_path):
            # Read model params from config.
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            model_name = 'gpt'
            inference_data_type = self.dtype
            model_args = dict(
                head_num=cfg.getint(model_name, 'head_num'),
                size_per_head=cfg.getint(model_name, "size_per_head"),
                layer_num=cfg.getint(model_name, "num_layer"),
                tensor_para_size=cfg.getint(model_name, "tensor_para_size"),
                vocab_size=cfg.getint(model_name, "vocab_size"),
                start_id=cfg.getint(model_name, "start_id"),
                end_id=cfg.getint(model_name, "end_id"),
                weights_data_type=cfg.get(model_name, "weight_data_type"),
                layernorm_eps=cfg.getfloat(model_name, 'layernorm_eps'),
                inference_data_type=inference_data_type)
        else:
            raise ValueError("Config.ini not found!")

        # update common parameters
        model_args.update(dict(
            lib_path=os.path.join(self.lib_path, "libth_transformer.so"),
            pipeline_para_size=self.pipeline_parallel_degree,
            int8_mode=0
        ))

        print('[FT][INFO] Load BLOOM model')
        for k, v in model_args.items():
            print(f' - {k.ljust(25, ".")}: {v}')

        # Check sanity and consistency between the model and tokenizer.
        checklist = ['head_num', 'size_per_head', 'vocab_size', 'layer_num',
                     'tensor_para_size', 'tensor_para_size', 'weights_data_type']
        if None in [model_args[k] for k in checklist]:
            none_params = [p for p in checklist if model_args[p] is None]
            print(f'[FT][WARNING] Found None parameters {none_params}. They must '
                  f'be provided either by config file or CLI arguments.')
        if model_args['start_id'] != self.tokenizer.bos_token_id:
            print('[FT][WARNING] Given start_id is not matched with the bos token '
                  'id of the pretrained tokenizer.')
        if model_args['end_id'] not in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id):
            print('[FT][WARNING] Given end_id is not matched with neither pad '
                  'token id nor eos token id of the pretrained tokenizer.')
        model = bloom.Bloom(**model_args)
        if not model.load(ckpt_path=ckpt_path):
            raise ValueError('No checkpoints are found')

        return model
