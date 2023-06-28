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
import configparser
import logging
import math
import os
import numpy as np

from transformers import T5Tokenizer, T5Config
from .examples.t5.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from .examples.t5.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from .ftmodel import InferenceModel
from .utils.common_utils import verify_and_convert


class T5Model(InferenceModel):

    def __init__(self, model: str,
                 tensor_parallel_degree: int,
                 pipeline_parallel_degree: int,
                 dtype: str,
                 **kwargs):
        super().__init__(model, tensor_parallel_degree, pipeline_parallel_degree, dtype, **kwargs)
        self.t5: FTT5 = None
        if self.dtype == "int8":
            raise NotImplementedError("T5 model does not support int8 mode!")

    def create_ft_model_artifacts(self, checkpoint_path):
        cmd = "CUDA_VISIBLE_DEVICES=-1 "
        cmd += f"python {os.path.dirname(os.path.realpath(__file__))}/examples/t5/huggingface_t5_ckpt_convert.py " \
               f"-i {self.model} -o {checkpoint_path}/ -i_g {self.num_gpus} -p {self.num_convert_process} " \
               f"-weight_data_type {self.weight_dtype}"
        file_string = [os.path.join(checkpoint_path, f'{self.num_gpus}-gpu/verify'), self.verify_str]
        verify_and_convert(cmd, file_string)

    def initialize(self):
        logging.info("Start model artifacts conversion...")
        self.create_ft_model_artifacts(self.model_dir)
        logging.info("load model...")
        self.build_t5_model()

    def build_t5_model(self):
        ckpt_config = configparser.ConfigParser()

        ckpt_config_path = os.path.join(self.model_dir, f'{self.num_gpus}-gpu', 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
        else:
            assert False, "[ERROR] This example only support loading model with FT format directly."

        weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
        encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                  d_model=ckpt_config.getint("encoder", "d_model"),
                                  d_kv=ckpt_config.getint("encoder", "d_kv"),
                                  d_ff=ckpt_config.getint("encoder", "d_ff"),
                                  num_layers=ckpt_config.getint("encoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("encoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                                  is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                                  )
        decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                  d_model=ckpt_config.getint("decoder", "d_model"),
                                  d_kv=ckpt_config.getint("decoder", "d_kv"),
                                  d_ff=ckpt_config.getint("decoder", "d_ff"),
                                  num_layers=ckpt_config.getint("decoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("decoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                                  decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                                  is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                                  )

        t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
        use_gated_activation = encoder_config.is_gated_act
        position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
        activation_type = encoder_config.feed_forward_proj

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1660
        # if tie_word_embeddings == True, scale the decoder output by sequence_output = sequence_output * (self.model_dim**-0.5)
        tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type
        )
        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )
        model_path = os.path.join(self.model_dir, f'{self.num_gpus}-gpu')
        ft_encoder_weight.load_from_bin(model_path, "Megatron")
        ft_decoding_weight.load_from_bin(model_path, "Megatron")

        if self.dtype == 'fp16':
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()
        elif self.dtype == 'bf16':
            ft_encoder_weight.to_bfloat16()
            ft_decoding_weight.to_bfloat16()

        target_lib = os.path.join(self.lib_path, "libth_transformer.so")
        q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
        remove_padding = True
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, target_lib, encoder_config.num_heads,
                                 encoder_config.d_kv, encoder_config.d_ff,
                                 encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                 encoder_config.relative_attention_num_buckets, 0,
                                 [],
                                 128, False, q_scaling, self.tensor_parallel_degree, self.pipeline_parallel_degree,
                                 False,
                                 position_embedding_type, moe_k=0,
                                 activation_type=activation_type, )
        ft_decoding = FTT5Decoding(ft_decoding_weight.w, target_lib,
                                   decoder_config.num_heads, decoder_config.d_kv,
                                   decoder_config.d_ff, encoder_config.d_model,
                                   decoder_config.d_model, decoder_config.num_layers,
                                   decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                   decoder_config.vocab_size,
                                   q_scaling,
                                   decoder_config.relative_attention_num_buckets, 0,
                                   [], max_distance=128,
                                   tensor_para_size=self.tensor_parallel_degree,
                                   pipeline_para_size=self.pipeline_parallel_degree,
                                   t5_with_bias=t5_with_bias,
                                   position_embedding_type=position_embedding_type, moe_k=0,
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
        self.get_tokenizer()
        input_tokens = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
        batch_token, batch_seq_len = self.generate(input_tokens, **kwargs)
        decoded_batch_token = []
        for j in range(len(batch_token)):
            decoded_batch_token.append(self.tokenizer.decode(batch_token[j][0][:batch_seq_len[j][0]],
                                                             skip_special_tokens=True))
        return decoded_batch_token
