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
import numpy as np

from transformers import T5ForConditionalGeneration, T5Tokenizer
from .examples.t5.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from .examples.t5.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from .ftmodel import InferenceModel

class T5Model(InferenceModel):
    STR_TO_NUMPY_TYPE_MAP = {"fp32": np.float32, "fp16": np.float16}

    def __init__(self, model: str,
                 tensor_parallel_degree: int,
                 pipeline_parallel_degree: int, dtype:str, **kwargs):
        super().__init__(model, tensor_parallel_degree, pipeline_parallel_degree, dtype, **kwargs)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model)
        self.t5: FTT5 = None
        self.batch_size = kwargs.get("batch_size", 1)

    def initialize(self):

        hf_model = T5ForConditionalGeneration.from_pretrained(self.model)

        # TODO: understand these
        encoder_config = hf_model.encoder.config
        decoder_config = hf_model.decoder.config
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
            weight_data_type=self.STR_TO_NUMPY_TYPE_MAP[self.dtype],
        )

        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
            t5_with_bias=False,
            use_gated_activation=False,
            t5_with_moe=False,
            position_embedding_type=0,
            weight_data_type=self.STR_TO_NUMPY_TYPE_MAP[self.dtype],
        )

        ft_encoder_weight.load_from_model(hf_model)
        ft_decoding_weight.load_from_model(hf_model)

        if self.dtype == 'fp16':
            hf_model = hf_model.half()
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()
        elif self.dtype == 'bf16':
            hf_model = hf_model  # bfloat inference not supported yet
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

        self.t5 = FTT5(ft_encoder, ft_decoding)

    def generate(self, input_tokens, **kwargs):
        default_args = dict(
            inputs_embeds=None,
            beam_width=1,
            max_seq_len=200,
            top_k=1,
            top_p=0.0,
            beam_search_diversity_rate=0.0,
            temperature=1.0,
            len_penalty=0.0,
            repetition_penalty=1.0,
            presence_penalty=None,
            min_length=0,
            random_seed=0,
            is_return_output_log_probs=False,
            is_return_cum_log_probs=False,
            is_return_cross_attentions=False,
            bad_words_list=None,
            stop_words_list=None
        )
        default_args.update(kwargs)
        default_args["beam_size"] = default_args["beam_width"]
        default_args.pop("beam_width")
        result = self.t5(input_tokens, **default_args)
        return result

    def pipeline_generate(self, inputs, **kwargs):
        total_iter = math.ceil(len(inputs) / self.batch_size)
        result = []
        output_tokens = []
        for it in range(total_iter):
            input_batch = inputs[it * self.batch_size: self.batch_size * (it + 1)]
            input_tokens = self.tokenizer(input_batch, return_tensors='pt', padding=True)
            ft_decoding_outputs, ft_decoding_seq_lens = self.generate(input_tokens)
            output_tokens.append((ft_decoding_outputs, ft_decoding_seq_lens))

        for batch_token, batch_seq_len in output_tokens:
            decoded_batch_token = []
            for j in range(len(batch_token)):
                decoded_batch_token.append(self.tokenizer.decode(batch_token[j][0][:batch_seq_len[j][0]], skip_special_tokens=True))
            result.append(decoded_batch_token)

        return result
