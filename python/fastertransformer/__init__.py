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
from transformers import AutoConfig

from .bloommodel import BLOOMModel
from .ftmodel import InferenceModel
from .optmodel import OPTModel
from .t5model import T5Model
from .gptmodel import GPTModel

# TODO: support commented model
SUPPORTED_MODEL_TYPES = {
    "t5": T5Model,
    "gpt2": GPTModel,
    # "gpt_neo": None,
    # "gptj": None,
    "opt": OPTModel,
    # "gpt_neox": None,
    "bloom": BLOOMModel,
}


def save_checkpoint(model: str,
                    tensor_parallel_degree: int,
                    pipeline_parallel_degree: int,
                    save_mp_checkpoint_path: str,
                    dtype: str = 'fp32',
                    **kwargs):
    inference_model = _get_inference_model(model,
                                           tensor_parallel_degree,
                                           pipeline_parallel_degree,
                                           dtype,
                                           is_mpi_mode=False,
                                           save_mp_checkpoint_path=save_mp_checkpoint_path,
                                           **kwargs)

    inference_model.create_ft_model_artifacts(save_mp_checkpoint_path)


def init_inference(model: str,
                   tensor_parallel_degree: int,
                   pipeline_parallel_degree: int,
                   dtype: str = 'fp32',
                   **kwargs):
    inference_model = _get_inference_model(model,
                                           tensor_parallel_degree,
                                           pipeline_parallel_degree,
                                           dtype,
                                           is_mpi_mode=True,
                                           save_mp_checkpoint_path=None,
                                           **kwargs)
    inference_model.initialize()
    return inference_model


def _get_inference_model(model: str,
                         tensor_parallel_degree: int,
                         pipeline_parallel_degree: int,
                         dtype: str = 'fp32',
                         is_mpi_mode: bool = True,
                         save_mp_checkpoint_path: str = None,
                         **kwargs):

    model_config = AutoConfig.from_pretrained(model)
    if model_config.model_type not in SUPPORTED_MODEL_TYPES.keys():
        raise ValueError(f"{model_config.model_type} type not supported for model {model}"
                         f"Supported model arch: {SUPPORTED_MODEL_TYPES.keys()}")

    return SUPPORTED_MODEL_TYPES[model_config.model_type](
        model, tensor_parallel_degree, pipeline_parallel_degree, dtype, is_mpi_mode, **kwargs)