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


class TritonModel:
    DEFAULT_TRITON_REPO = "/tmp/fastertransformer/all_models"

    def __init__(self, model, model_type):
        self.model = model
        self.tokenizer = model.tokenizer
        self.model_type = model_type
        if model_type in ["gpt2", "opt"]:
            self.model_type = "gpt"
        self.base_name = re.sub(r'\W+', '-', self.model.model)
        self.model_dir = self.DEFAULT_TRITON_REPO + "/" + self.base_name
        self.core = None
        self.predictor = None

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
        # TODO: Allow stream inputs
        self.fix_config(config_path, self.model.model_dir, self.model.tensor_parallel_degree, None)
        logging.info("Converting completed, start loading...")
        self.core = tritontoolkit.init_triton(self.DEFAULT_TRITON_REPO)
        self.predictor = self.core.load_model(self.base_name)

    def generate(self):
        pass

    def pipeline_generate(self, inputs, output_length):
        pass

    def stream_generate(self):
        pass

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
