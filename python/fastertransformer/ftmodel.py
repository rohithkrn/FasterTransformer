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
import tempfile
import torch.distributed as dist
from transformers import AutoTokenizer

class InferenceModel:
    DEFAULT_LIB_PATH = "/opt/tritonserver/backends/fastertransformer"
    DEFAULT_SAVE_DIR = os.path.join(tempfile.gettempdir(), "ft_model")

    def __init__(self, model: str, tensor_parallel_degree, pipeline_parallel_degree, dtype="fp32",
                 **kwargs):
        logging.info("Initializing inference model with FasterTransformer")
        self.model = model
        self.tokenizer = None
        self.tensor_parallel_degree = tensor_parallel_degree
        self.pipeline_parallel_degree = pipeline_parallel_degree
        self.set_data_type(dtype)
        self.lib_path = kwargs.get("lib_path", self.DEFAULT_LIB_PATH)
        self.verify_str = f"{self.model}-{self.weight_dtype}-{self.tensor_parallel_degree}-{self.pipeline_parallel_degree}"
        self.num_convert_process = kwargs.get("num_convert_process", 8)
        self.num_gpus = tensor_parallel_degree * pipeline_parallel_degree

        if os.getenv('OMPI_COMM_WORLD_SIZE'):
            dist.init_process_group("mpi")

        self.model_dir = self.model if os.path.exists(self.model) else self.DEFAULT_SAVE_DIR
        # Check model if partitioned
        verify_path = os.path.join(self.model, f'{self.num_gpus}-gpu/verify')
        if os.path.exists(verify_path):
            with open(verify_path, "r") as f:
                self.verify_str = f.readlines()[0]

    def set_data_type(self, dtype):
        self.dtype = dtype
        if dtype == "fp16" or dtype == "int8":
            self.weight_dtype = "fp16"
        elif dtype == "fp32" or dtype == "bf16":
            self.weight_dtype = "fp32"
        else:
            raise NotImplementedError(f"Not implemented for {dtype}!")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        return self.tokenizer


    def initialize(self):
        raise NotImplementedError("Method not implemented for InferenceModel")

    def generate(self, **kwargs):
        raise NotImplementedError("Method not implemented for InferenceModel")

    def pipeline_generate(self, **kwargs):
        raise NotImplementedError("Method not implemented for InferenceModel")

    def create_ft_model_artifacts(self, checkpoint_path):
        raise NotImplementedError("Method not implemented for InferenceModel")
