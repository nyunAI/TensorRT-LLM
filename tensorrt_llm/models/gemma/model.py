# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

from ..._utils import pad_vocab_size
from ...functional import Tensor, recv, send
from ...layers import (Attention, AttentionMaskType, ColumnLinear, Embedding,
                       GatedMLP, PositionEmbeddingType, PromptTuningEmbedding,
                       RmsNorm)
from ...mapping import Mapping
from ...module import Module
from ...quantization import QuantMode
from ...top_model_mixin import TopModelMixin
from ..modeling_utils import DecoderLayerList, DecoderModelForCausalLM
from .weight import load_from_hf_gemma


class GemmaDecoderLayer(Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attention_head_size=config.head_size,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
        )

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            ffn_hidden_size=mlp_hidden_size,
                            hidden_act=config.hidden_act,
                            dtype=config.dtype,
                            bias=config.mlp_bias,
                            tp_group=config.mapping.tp_group,
                            tp_size=config.mapping.tp_size,
                            quant_mode=config.quant_mode)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            medusa_packed_mask=None,  # For Medusa support
            medusa_position_offsets=None,
            use_cache=False,
            kv_cache_params=None,
            attention_params=None,
            lora_layer_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            medusa_packed_mask=medusa_packed_mask,  # For Medusa support
            medusa_position_offsets=medusa_position_offsets,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GemmaModel(Module):

    def __init__(self, config) -> None:
        super().__init__()

        self.mapping = config.mapping
        self.use_prompt_tuning = config.use_prompt_tuning
        EmbeddingCls = PromptTuningEmbedding if config.use_prompt_tuning else Embedding
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = EmbeddingCls(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                dtype=config.dtype,
                tp_size=self.mapping.tp_size
                if config.use_parallel_embedding else 1,
                tp_group=self.mapping.tp_group
                if config.use_parallel_embedding else None,
                sharding_dim=config.embedding_sharding_dim,
                tp_rank=self.mapping.tp_rank,
            )

        self.layers = DecoderLayerList(GemmaDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if self.use_prompt_tuning else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_params=lora_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GemmaForCausalLM(DecoderModelForCausalLM, TopModelMixin):

    def __init__(self, config):

        self.check_config(config)
        transformer = GemmaModel(config)

        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping

        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(cls,
                          hf_model_dir,
                          dtype='float16',
                          mapping: Optional[Mapping] = None,
                          quant_mode: Optional[QuantMode] = None,
                          **kwargs):
        import transformers
        from transformers import GemmaConfig

        from ...models.modeling_utils import PretrainedConfig
        cfg = GemmaConfig.from_pretrained(hf_model_dir)

        num_kv_heads = cfg.num_key_value_heads if hasattr(cfg, "num_key_value_heads") \
            else cfg.num_attention_heads
        if mapping is None:
            mapping = Mapping()
        if quant_mode is None:
            quant_mode = QuantMode(0)

        cfg.mapping = mapping

        cfg.dtype = dtype
        cfg.quant_mode = quant_mode

        cfg.norm_epsilon = cfg.rms_norm_eps

        config = {
            'architecture': cfg.architectures[0],
            'dtype': cfg.dtype,
            'logits_dtype': 'float32',
            'num_hidden_layers': cfg.num_hidden_layers,
            'num_attention_heads': cfg.num_attention_heads,
            'head_size': cfg.head_dim,
            'hidden_size': cfg.hidden_size,
            'intermediate_size': cfg.intermediate_size,
            'num_key_value_heads': cfg.num_key_value_heads,
            'vocab_size': cfg.vocab_size,
            'position_embedding_type': 'rope_gpt_neox',
            'max_position_embeddings': cfg.max_position_embeddings,
            'hidden_act': cfg.hidden_act,
            'rotary_base': getattr(cfg, 'rotary_base', 10000.0),
            'rotary_scaling': getattr(cfg, 'rotary_scaling', None),
            'norm_epsilon': cfg.rms_norm_eps,
            'quantization': quant_mode.to_dict(),
            'mapping': {
                'world_size': mapping.world_size,
                'tp_size': mapping.world_size,
            },
            'use_parallel_embedding': kwargs.get("use_parallel_embedding",
                                                 False),
            'embedding_sharding_dim': kwargs.get("embedding_sharding_dim", 0),
            'use_prompt_tuning': kwargs.get("use_prompt_tuning", False),
            'use_fused_mlp': kwargs.get("use_fused_mlp", False),
        }

        assert not quant_mode.has_any_quant()

        tllm_llama = GemmaForCausalLM(PretrainedConfig.from_dict(config))

        hf_model = transformers.GemmaForCausalLM
        hf_llama = hf_model.from_pretrained(
            hf_model_dir,
            device_map={
                "model": "cpu",
                "lm_head": "cpu",
                "embed_tokens": "cpu",
                "layers": "cpu",
                "norm": "cpu",
            },  # Load to CPU memory
            torch_dtype='auto',
        )

        weights = load_from_hf_gemma(
            tllm_llama,
            hf_llama,
            mapping=mapping,
            dtype=dtype,
            # TODO: these shall be outside from_hugging_face too.
            use_gemm_woq_plugin=kwargs.get("use_gemm_woq_plugin", False),
        )
        del hf_llama
        tllm_llama.load(weights)
        return tllm_llama

    def check_config(self, config):
        config.set_if_not_exist('use_parallel_embedding', False)
        config.set_if_not_exist('embedding_sharding_dim', 0)
        config.set_if_not_exist('mlp_bias', False)
        config.set_if_not_exist('attn_bias', False)
        config.set_if_not_exist('rotary_base', 10000.0)
        config.set_if_not_exist('rotary_scaling', None)
        config.set_if_not_exist('use_fused_mlp', False)
