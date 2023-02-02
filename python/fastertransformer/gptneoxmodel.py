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
from .ftmodel import InferenceModel
from torch import nn


class Args:
    saved_dir: str
    in_file: str
    model_name: str
    infer_gpu_num: int
    weight_data_type: str
    trained_gpu_num = 1
    prompt_in_file_list: None
    processes = 4


class GPTNeoX(InferenceModel):
    # TODO: optimize this
    DEFAULT_SAVING_LOC = "/opt/djl/ft_model/"

    def initialize(self):
        args = Args()
        args.in_file = self.model
        args.infer_gpu_num = self.tensor_parallel_degree
        args.weight_data_type = self.dtype
        args.saved_dir = self.DEFAULT_SAVING_LOC
        raise NotImplementedError("Not implemented")


class FTGPTNeoX(nn.Module):
    def __init__(self, lib_path, saved_model_path):
        super().__init__()
        # FIXME: Not implemented
        pass
