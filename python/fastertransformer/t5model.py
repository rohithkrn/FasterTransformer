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
import math
import os

from .examples.t5.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from .examples.t5.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from .ftmodel import InferenceModel
import torch.distributed as dist


class T5Model(InferenceModel):

    def initialize(self):
        if dist.is_mpi_available():
            try:
                dist.init_process_group(backend='mpi')
                rank = dist.get_rank()
            except:
                rank = dist.get_rank()
        else:
            rank = 0

        # TODO: understand these
        encoder_config = self.model.encoder.config
        decoder_config = self.model.decoder.config
        encoder_config.update({"num_experts": 0})
        decoder_config.update({"num_experts": 0})
        encoder_config.update({"moe_layer_index": []})
        decoder_config.update({"moe_layer_index": []})

        activation_type = encoder_config.feed_forward_proj
        tie_word_embeddings = decoder_config.tie_word_embeddings
        q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))

        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
            t5_with_bias=False,
            use_gated_activation=False,
            t5_with_moe=False,
            position_embedding_type=0,
            weight_data_type=self.dtype,
        )

        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
            t5_with_bias=False,
            use_gated_activation=False,
            t5_with_moe=False,
            position_embedding_type=0,
            weight_data_type=self.dtype,
        )

        ft_encoder_weight.load_from_model(self.model)
        ft_decoding_weight.load_from_model(self.model)

        if self.dtype == 'fp16':
            self.model = self.model.half()
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()
        elif self.dtype == 'bf16':
            self.model = self.model  # bfloat inference not supported yet
            ft_encoder_weight.to_bfloat16()
            ft_decoding_weight.to_bfloat16()

        target_lib = os.path.join(self.lib_path, "libth_transformer.so")
        remove_padding = True if self.batch_size > 32 else False
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, target_lib, encoder_config.num_heads,
                                 encoder_config.d_kv, encoder_config.d_ff,
                                 encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                 encoder_config.relative_attention_num_buckets, encoder_config.num_experts,
                                 encoder_config.moe_layer_index,
                                 128, False, q_scaling, self.tensor_parallel_degree, self.pipeline_parallel_degree,
                                 False,
                                 0, moe_k=0,
                                 activation_type=activation_type, )
        ft_decoding = FTT5Decoding(ft_decoding_weight.w, target_lib,
                                   decoder_config.num_heads, decoder_config.d_kv,
                                   decoder_config.d_ff, encoder_config.d_model,
                                   decoder_config.d_model, decoder_config.num_layers,
                                   decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                   decoder_config.vocab_size,
                                   q_scaling,
                                   decoder_config.relative_attention_num_buckets, decoder_config.num_experts,
                                   decoder_config.moe_layer_index, max_distance=128,
                                   tensor_para_size=self.tensor_parallel_degree,
                                   pipeline_para_size=self.pipeline_parallel_degree,
                                   t5_with_bias=False,
                                   position_embedding_type=0, moe_k=0,
                                   activation_type=activation_type, tie_word_embeddings=tie_word_embeddings, )

        self.model = FTT5(ft_encoder, ft_decoding)
