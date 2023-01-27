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
import torch
from .ftmodel import InferenceModel
from .t5model import T5Model

SUPPORTED_MODEL_TYPES = {
    "t5": T5Model,
    "gpt2": None,
    "gpt_neo": None,
    "gptj": None,
    "opt": None,
    "gpt_neox": None,
    "bloom": None,
}


def init_inference(model: torch.nn.Module,
                   tensor_parallel_degree: int,
                   pipeline_parallel_degree: int,
                   **kwargs):
    if model.config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"{model.config.model_type} not supported! "
                         f"Supported model arch: {SUPPORTED_MODEL_TYPES}")
    inference_model = SUPPORTED_MODEL_TYPES[model.config.model_type] \
        (model, tensor_parallel_degree, pipeline_parallel_degree, **kwargs)
    inference_model.initialize()
    return inference_model
