# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from ..tensor import (
    Optimizer,
    QBytesTensor,
    qtype,
    quantize_activation,
    quantize_weight,
)
from .qmodule import QAttentionModuleMixin, register_qmodule

__all__ = ["QMultiheadAttention"]


@register_qmodule(torch.nn.MultiheadAttention)
class QMultiheadAttention(QAttentionModuleMixin, torch.nn.MultiheadAttention):
    @classmethod
    def qcreate(
        cls,
        module,
        weights: qtype,
        activations: Optional[qtype] = None,
        optimizer: Optional[Optimizer] = None,
    ):
        return cls(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            dropout=module.dropout,
            bias=True if module.in_proj_bias is not None else False,
            add_bias_kv=True if module.bias_k is not None else False,
            add_zero_attn=module.add_zero_attn,
            kdim=module.kdim,
            vdim=module.vdim,
            batch_first=module.batch_first,
            device=module.in_proj_weight.device,
            dtype=module.in_proj_weight.dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
        )

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        # Only support self attention
        if self.activation_qtype is not None and not isinstance(input, QBytesTensor):
            # Quantize activations to be able to take advantage of accelerated matmul
            input = quantize_activation(input, qtype=self.activation_qtype, scale=self.input_scale)
        # We always use quantized weights, but do not quantize the bias
        return torch.nn.functional.multi_head_attention_forward(
            query=input,
            key=input,
            value=input,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=quantize_weight(
                self.in_proj_weight,
                qtype=self.weight_qtype,
                axis=0,
                group_size=self.weight_group_size,
                optimizer=self.optimizer,
            ),
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=quantize_weight(
                self.out_proj.weight,
                qtype=self.weight_qtype,
                axis=0,
                group_size=self.weight_group_size,
                optimizer=self.optimizer,
            ),
            out_proj_bias=self.out_proj.bias,
        )[0]
