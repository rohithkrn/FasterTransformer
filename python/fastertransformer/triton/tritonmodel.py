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
import logging
import os
import shutil
import re
import tritontoolkit
import sys
import numpy as np


class TritonModel:
    DEFAULT_TRITON_REPO = "/tmp/fastertransformer/all_models"

    NUMPY_DTYPE_MAPPER = {
        "BOOL": np.bool_,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        'UINT64': np.uint64,
        "INT8": np.int8,
        "INT32": np.int32,
        "INT64": np.int64,
        'FP16': np.float16,
        'FP32': np.float32,
        'FP64': np.float64,
    }

    def __init__(self, model, model_type, do_streaming=False):
        self.model = model
        self.tokenizer = model.tokenizer
        self.model_type = model_type
        if model_type in ["gpt2", "opt"]:
            self.model_type = "gpt"
        self.base_name = re.sub(r'\W+', '-', self.model.model)
        self.model_dir = self.DEFAULT_TRITON_REPO + "/" + self.base_name
        self.core = None
        self.predictor = None
        self.do_streaming = do_streaming
        self.input_info = {}

    def initialize(self):
        logging.info("Converting hf model to ft model...")
        self.model.create_ft_model_artifacts(self.model.model_dir)
        pkgdir = sys.modules['fastertransformer'].__path__[0]
        config_path = os.path.join(self.model_dir, "config.pbtxt")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir + "/1")
        src_pbtxt = os.path.join(pkgdir, f"triton/all_models/{self.model_type}/config.pbtxt")
        if not os.path.exists(src_pbtxt):
            raise NotImplementedError(f"{self.model_type} is not supported by triton core!")
        shutil.copy(src_pbtxt, config_path)
        self.fix_config(config_path, self.model.model_dir, self.model.tensor_parallel_degree, self.do_streaming)
        logging.info("Converting completed, start loading...")
        self.core = tritontoolkit.init_triton(self.DEFAULT_TRITON_REPO)
        self.predictor = self.core.load_model(self.base_name)
        self.compute_input_info(self.predictor.metadata)

    def compute_input_info(self, metadata):
        for element in metadata['inputs']:
            self.input_info[element['name']] = {'dtype': self.NUMPY_DTYPE_MAPPER[element['datatype']],
                                                'shape': element['shape']}

    def get_input_options(self):
        return self.input_info

    def generate(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.input_info.keys():
                raise AssertionError(f"{key} is not in available inputs {self.input_info.keys()}")
            if value.dtype != self.input_info[key]['dtype']:
                raise AssertionError(f"Data type mismatch, required {self.input_info[key]['dtype']},"
                                     f" actual {value.dtype}")
        return self.predictor.inference(kwargs)

    def pipeline_generate(self, inputs: list, output_length: list, padding_side="right", **kwargs):
        final_kwargs, _ = self.input_builder(inputs, output_length, kwargs, padding_side=padding_side)
        result = self.generate(**final_kwargs)
        return self.tokenizer.batch_decode(result['output_ids'].squeeze(1), skip_special_tokens=True)

    def stream_generate(self, inputs: list, output_length: list, padding_side="right", **kwargs):
        final_kwargs, offset = self.input_builder(inputs, output_length, kwargs, padding_side=padding_side)
        gen = self.predictor.stream_inference(final_kwargs)
        prev = None
        for data in gen:
            sequence_lengths = data["sequence_length"].squeeze(1) - offset
            output_ids = data["output_ids"]
            if prev is None:
                prev = [0] * len(sequence_lengths)
            final_tokens = []
            for i in range(len(sequence_lengths)):
                tokens = self.tokenizer.decode(output_ids[i][0][prev[i]:sequence_lengths[i]],
                                               skip_special_tokens=True)
                final_tokens.append(tokens)
            prev = sequence_lengths
            yield final_tokens
        return 0

    def input_builder(self, inputs: list, output_length: list, kwargs, padding_side="left"):
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        seq_length = []
        token_lines = []
        for line in inputs:
            input_ids = self.tokenizer(line).input_ids
            seq_length.append(len(input_ids))
            token_lines.append(input_ids)

        kwargs["input_ids"], max_len = self.stack_padding(token_lines, padding_side, self.tokenizer.pad_token_id)

        if self.model_type == "t5":
            kwargs["max_output_len"] = output_length
            kwargs["sequence_length"] = seq_length
        else:
            kwargs["request_output_len"] = output_length
            kwargs["input_lengths"] = seq_length
        final_kwargs = {}
        for key, value in kwargs.items():
            required_dtype = self.input_info[key]['dtype']
            required_shape = len(self.input_info[key]['shape'])
            if key not in self.input_info.keys():
                raise AssertionError(f"{key} is not in available inputs {self.input_info.keys()}")
            final_kwargs[key] = np.array(value, dtype=required_dtype)
            if required_shape != final_kwargs[key].ndim:
                if required_shape < final_kwargs[key].ndim:
                    raise AssertionError(f"Shape mismatch for {key}, required {required_shape}"
                                         f" , given {final_kwargs[key].ndim}")
                final_kwargs[key] = \
                    final_kwargs[key].reshape((-1,) + (1,) * (required_shape - final_kwargs[key].ndim))
        return final_kwargs, max_len - np.array(seq_length)

    def stack_padding(self, nds, side, value):
        max_len = max([len(arr) for arr in nds])
        if "left" == side:
            return np.array(
                [np.pad(arr, (max_len - len(arr), 0), 'constant', constant_values=value) for arr in nds]), max_len
        if "right" == side:
            return np.array(
                [np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=value) for arr in nds]), max_len

    def fix_config(self, config_path, model_dir, tp, do_stream=None):
        with open(config_path, "r") as f:
            configs = f.readlines()
            for idx, line in enumerate(configs):
                if do_stream and "decoupled" in line:
                    configs[idx] = "  decoupled: True\n"
                if line.startswith("name:"):
                    base_name = os.path.basename(os.path.dirname(config_path))
                    configs[idx] = f'name: "{base_name}"\n'
                if "tensor_para_size" in line:
                    configs[idx + 2] = f'    string_value: "{tp}"\n'
                if "model_checkpoint_path" in line:
                    configs[idx + 2] = f'    string_value: "{model_dir}/{tp}-gpu/"\n'
        with open(config_path, "w") as f:
            f.writelines(configs)
