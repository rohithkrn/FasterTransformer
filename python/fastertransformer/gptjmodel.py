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

from .gptmodel import GPTModel
from .utils.common_utils import verify_and_convert
import os


class GPTJModel(GPTModel):

    def initialize(self):
        raise NotImplementedError("Not supported under python mode")

    def generate(self, **kwargs):
        raise NotImplementedError("Not supported under python mode")

    def create_ft_model_artifacts(self, checkpoint_path):
        cmd = "CUDA_VISIBLE_DEVICES=-1 "
        cmd += f"python {os.path.dirname(os.path.realpath(__file__))}/examples/gptj/huggingface_gptj_convert.py " \
               f"--model-dir {self.model} --output-dir {checkpoint_path}" \
               f" --n-inference-gpus {self.num_convert_process} --n-inference-gpus {self.num_gpus}"
        file_string = [os.path.join(checkpoint_path, f'{self.num_gpus}-gpu/verify'), self.verify_str]
        verify_and_convert(cmd, file_string)
