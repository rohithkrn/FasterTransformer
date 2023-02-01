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
import numpy as np
import logging


class InferenceModel:
    DEFAULT_LIB_PATH = "/usr/local/backends/fastertransformer"

    def __init__(self, model: str, tensor_parallel_degree, pipeline_parallel_degree, dtype=np.float32,
                 batch_size=1, **kwargs):
        logging.info("Initializing inference model with FasterTransformer")
        self.model = model
        self.tensor_parallel_degree = tensor_parallel_degree
        self.pipeline_parallel_degree = pipeline_parallel_degree
        self.dtype = dtype
        self.batch_size = batch_size
        self.lib_path = self.DEFAULT_LIB_PATH if "lib_path" not in kwargs else kwargs["lib_path"]

    def initialize(self):
        raise NotImplementedError("Method not implemented for InferenceModel")

    def generate(self):
        raise NotImplementedError("Method not implemented for InferenceModel")

    def create_ft_model_artifacts(self):
        raise NotImplementedError("Method not implemented for InferenceModel")