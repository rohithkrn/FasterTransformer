#!/bin/bash
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
# copy t5 source scripts to toolkit directory
mkdir -p fastertransformer/examples/{t5,gptneox,gpt}
echo "" > fastertransformer/examples/t5/__init__.py
echo "" > fastertransformer/examples/gptneox/__init__.py
echo "" > fastertransformer/examples/gpt/__init__.py
echo "" > fastertransformer/examples/__init__.py
cp ../examples/pytorch/t5/utils/{ft_encoder.py,ft_decoding.py,huggingface_t5_ckpt_convert.py} fastertransformer/examples/t5/
cp ../examples/pytorch/gptneox/utils/{gptneox.py,huggingface_gptneox_convert.py} fastertransformer/examples/gptneox/
cp ../examples/pytorch/gpt/utils/{gpt_token_encoder.py,gpt.py,parallel_gpt.py,comm.py,huggingface_gpt_convert.py,huggingface_opt_convert.py} fastertransformer/examples/gpt/
cp ../examples/pytorch/gpt/utils/{huggingface_bloom_convert.py,bloom.py} fastertransformer/examples/gpt/
