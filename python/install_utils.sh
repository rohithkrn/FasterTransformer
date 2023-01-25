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
mkdir -p fastertransformer/examples/t5
echo "" > fastertransformer/examples/t5/__init__.py
echo "" > fastertransformer/examples/__init__.py
cp ../examples/pytorch/t5/utils/ft_encoder.py fastertransformer/examples/t5/
cp ../examples/pytorch/t5/utils/ft_decoding.py fastertransformer/examples/t5/
