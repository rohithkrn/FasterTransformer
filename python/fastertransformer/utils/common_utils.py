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
import os.path
import os.listdir
import subprocess
import logging
import torch.distributed as dist
import shutil
from huggingface_hub import snapshot_download, HfApi
from pathlib import Path


def verify_and_convert(command: str, file_string):

    # If there are parallel processes, convert only at rank 0
    logging.info(f"File string {file_string}")
    if not dist.is_initialized() or dist.get_rank() == 0:
        found = False
        if os.path.exists(file_string[0]):
            with open(file_string[0], "r") as f:
                if file_string[1] == f.readlines()[0].strip():
                    found = True
                else:
                    shutil.rmtree(os.path.dirname(file_string[0]))  # remove dir if something is there
        if not found:
            logging.debug(f"executing command {command}")
            # FIXME: Process will hang if convert failed
            subprocess.check_call(command, shell=True)
            with open(file_string[0], "w") as f:
                f.write(file_string[1])
    if dist.is_initialized():
        dist.barrier()


def execute_command_with_rank(command: str, rank):
    if rank == 0:
        logging.debug(f"executing command {command}")
        subprocess.check_call(command, shell=True)
    if dist.is_initialized():
        dist.barrier()


def download_model(model_id, download_dir="/tmp/hf_downloaded_model"):
    if not dist.is_initialized() or dist.get_rank() == 0:
        if os.path.exists(model_id):
            return model_id
        local_model_path = Path(download_dir)
        local_model_path.mkdir(exist_ok=True)
        model_name = model_id
        hf_api = HfApi()
        model_files = hf_api.list_files_info(model_name)
        safetensor_files = [f for f in model_files if f.rfilename.endswith(".safetensors")]
        # Only download pytorch checkpoint files
        allow_patterns = ["*.json", "*.txt", "*.model"]
        if len(safetensor_files) > 0:
            allow_patterns += ["*.safetensors"]
        else:
            allow_patterns += ["*.bin"]

        # - Leverage the snapshot library to download the model since the model is stored in repository using LFS
        return snapshot_download(
            repo_id=model_name,
            cache_dir=local_model_path,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False
        )
    return None
