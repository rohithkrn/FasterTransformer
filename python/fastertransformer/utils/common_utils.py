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
import subprocess
import logging
import torch.distributed as dist


def execute_command(command: str, rank):
    if rank == 0:
        logging.debug(f"executing command {command}")
        subprocess.check_call(command, shell=True)
    if dist.is_initialized():
        dist.barrier()
