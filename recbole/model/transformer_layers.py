import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_
from einops import rearrange, repeat
from functools import partial

from recbole.utils import FeatureType, FeatureSource
from recbole.model.layers import FeedForward, MultiHeadAttention


### MLPMixer ###
class MLPMixerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        hidden_size=64,
        seq_len=50,
        inner_size=256,
        hidden_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(MLPMixerEncoder, self).__init__()

        layer = MLPMixerLayer(
            hidden_size,
            seq_len,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MLPMixerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(MLPMixerLayer, self).__init__()
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.expansion_factor = intermediate_size // hidden_size
        self.pre1 = PreNormResidual(
            hidden_size,
            FlexFeedForward(
                seq_len,
                self.expansion_factor,
                hidden_dropout_prob,
                hidden_act,
                layer_norm_eps,
                self.chan_first,
            ),
            layer_norm_eps,
        )
        self.pre2 = PreNormResidual(
            hidden_size,
            FlexFeedForward(
                hidden_size,
                self.expansion_factor,
                hidden_dropout_prob,
                hidden_act,
                layer_norm_eps,
                self.chan_last,
            ),
            layer_norm_eps,
        )

    def forward(self, hidden_states):
        item_mixer_output = self.pre1(hidden_states)
        channel_mixer_output = self.pre2(item_mixer_output)
        return channel_mixer_output


class PreNormResidual(nn.Module):
    def __init__(self, hidden_size, ffn, layer_norm_eps):
        super(PreNormResidual, self).__init__()
        self.ffn = ffn
        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        return self.ffn(self.ln(hidden_states)) + hidden_states


class FlexFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        expansion_factor,
        dropout_prob,
        hidden_act,
        layer_norm_eps,
        dense=nn.Linear,
    ):
        super(FlexFeedForward, self).__init__()
        self.dense_1 = dense(dim, dim * expansion_factor)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = dense(dim * expansion_factor, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


### gMLP ###
class gMLPEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        hidden_size=64,
        seq_len=50,
        inner_size=256,
        hidden_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(gMLPEncoder, self).__init__()
        layer = gMLPLayer(
            hidden_size,
            seq_len,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class gMLPLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(gMLPLayer, self).__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.proj_1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(intermediate_size, seq_len)
        self.proj_2 = nn.Linear(intermediate_size // 2, hidden_size)

    def forward(self, hidden_states):
        shorcut = hidden_states.clone()
        hidden_states = self.ln(hidden_states)
        hidden_states = self.proj_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.spatial_gating_unit(hidden_states)
        hidden_states = self.proj_2(hidden_states)
        return hidden_states + shorcut


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super(SpatialGatingUnit, self).__init__()
        self.ln = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.ln(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v


### Linformer ###
class LinearMap(nn.Module):
    def __init__(self, seq_len, k_heads):
        super().__init__()
        self.k_heads = k_heads  # k head
        # self.theta_k = nn.Parameter(torch.randn([self.k_heads, seq_len]))
        self.lin = nn.Linear(seq_len, self.k_heads, bias=True)
        torch.nn.init.xavier_normal_(self.lin.weight)

    def forward(self, input_tensor):  # [B, L, d] -> [B, k, d]
        # result = torch.matmul(self.theta_k, input_tensor)
        result = self.lin(input_tensor.transpose(1, 2))
        return result.transpose(1, 2)


class LinformerAttention(nn.Module):
    def __init__(
        self,
        config,
        n_heads,
        seq_len,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LinformerAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )
        k_heads = config["k_heads"]
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)  # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.linear_key = LinearMap(seq_len, k_heads)
        self.linear_value = LinearMap(seq_len, k_heads)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask=None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.linear_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(self.linear_value(mixed_value_layer))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # [1024, 2, 50, 50]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores  # + attention_mask #+ abs_pos_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [1024, 2, 50, 32]
        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()  # [1024, 50, 2, 32]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )  # [1024, 50, 64]
        context_layer = context_layer.view(*new_context_layer_shape)  # [1024, 50, 64]

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


### Synthesizer ###
class SynthesizerAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        seq_len,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(SynthesizerAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )
        self.seq_len = seq_len

        self.k = int(0.1 * self.seq_len)

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.k)
        self.key = nn.Linear(hidden_size, self.k)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)  # [1024, 50, 64]
        mixed_value_layer = self.value(input_tensor)

        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            mixed_query_layer, mixed_key_layer.transpose(-1, -2)
        )

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        attention_probs = attention_probs.unsqueeze(1)
        context_layer = torch.matmul(attention_probs, value_layer)  # [1024, 2, 50, 32]
        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()  # [1024, 50, 2, 32]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )  # [1024, 50, 64]
        context_layer = context_layer.view(*new_context_layer_shape)  # [1024, 50, 64]

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


### LinearTrm ###
class LinearTrmAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        seq_len,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LinearTrmAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )
        self.eps = 1e-6
        self.seq_len = seq_len

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)  # [1024, 50, 64]
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [1024, 50, 2, 32]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_layer = nn.functional.elu(query_layer) + 1
        key_layer = nn.functional.elu(key_layer) + 1

        # query_layer = query_layer.contiguous().permute(0, 2, 1, 3)#[1024, 2, 50, 32]
        # key_layer = key_layer.contiguous().permute(0, 2, 1, 3)
        # value_layer = value_layer.contiguous().permute(0, 2, 1, 3)

        # key_value = torch.einsum('...sd,...se->...de', key_layer, value_layer)
        # result = 1.0 / torch.einsum('...sd,...d->...s', query_layer, key_layer.sum(dim=-2) + self.eps)
        # context_layer = torch.einsum('...de,...sd,...s->...se', key_value, query_layer, result)

        key_value = torch.einsum("nshd,nshm->nhmd", key_layer, value_layer)
        result = 1.0 / torch.einsum(
            "nlhd,nhd->nlh", query_layer, key_layer.sum(dim=1) + self.eps
        )
        context_layer = torch.einsum(
            "nlhd,nhmd,nlh->nlhm", query_layer, key_value, result
        )
        context_layer = context_layer.contiguous()

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #[1024, 50, 2, 32]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )  # [1024, 50, 64]
        context_layer = context_layer.view(*new_context_layer_shape)  # [1024, 50, 64]

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


### Performer ###
class PerformerAttention(nn.Module):
    def __init__(
        self,
        config,
        n_heads,
        seq_len,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(PerformerAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )
        ortho_scaling = config["ortho_scaling"]
        qr_uniform_q = config["qr_uniform_q"]
        self.eps = 1e-6
        self.seq_len = seq_len

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # performer part
        self.nb_features = int(
            config["nb_features_ratio"] * self.seq_len
        )  # int(self.attention_head_size * math.log(self.attention_head_size))
        self.ortho_scaling = ortho_scaling

        self.create_projection_q = partial(
            self.gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=self.attention_head_size,
            scaling=ortho_scaling,
            qr_uniform_q=qr_uniform_q,
        )
        self.create_projection_k = partial(
            self.gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=self.attention_head_size,
            scaling=ortho_scaling,
            qr_uniform_q=qr_uniform_q,
        )
        self.projection_matrix_q = self.create_projection_q()
        self.projection_matrix_k = self.create_projection_k()

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def orthogonal_matrix_chunk(self, cols, qr_uniform_q=False, device=None):
        unstructured_block = torch.randn((cols, cols), device=device)
        q, r = torch.qr(unstructured_block.cpu(), some=True)
        q, r = map(lambda t: t.to(device), (q, r))

        # proposed by @Parskatt
        # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
        if qr_uniform_q:
            d = torch.diag(r, 0)
            q *= d.sign()
        return q.t()

    def gaussian_orthogonal_random_matrix(
        self, nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None
    ):
        nb_full_blocks = int(nb_rows / nb_columns)  # 5
        block_list = []

        for _ in range(nb_full_blocks):
            q = self.orthogonal_matrix_chunk(
                nb_columns, qr_uniform_q=qr_uniform_q, device=device
            )
            block_list.append(q)

        remaining_rows = nb_rows - nb_full_blocks * nb_columns
        if remaining_rows > 0:
            q = self.orthogonal_matrix_chunk(
                nb_columns, qr_uniform_q=qr_uniform_q, device=device
            )
            block_list.append(q[:remaining_rows])

        final_matrix = torch.cat(block_list)

        if scaling == 0:
            multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
        elif scaling == 1:
            multiplier = math.sqrt((float(nb_columns))) * torch.ones(
                (nb_rows,), device=device
            )
        else:
            raise ValueError(f"Invalid scaling {scaling}")

        a = torch.diag(multiplier) @ final_matrix
        return a  # [160, 32]

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def softmax_kernel(
        self,
        data,
        *,
        projection_matrix,
        is_query,
        normalize_data=True,
        eps=1e-4,
        device=None,
    ):
        b, h, *_ = data.shape

        data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

        ratio = projection_matrix.shape[0] ** -0.5

        projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
        projection = projection.type_as(data)
        # print('projection', projection.size())

        data_dash = torch.einsum(
            "...id,...jd->...ij", (data_normalizer * data), projection
        )
        # print('data_dash1', data_dash.size())
        diag_data = data**2
        diag_data = torch.sum(diag_data, dim=-1)
        diag_data = (diag_data / 2.0) * (data_normalizer**2)
        diag_data = diag_data.unsqueeze(dim=-1)
        # print('diag_data', diag_data.size())

        if is_query:
            data_dash = ratio * (
                torch.exp(
                    data_dash
                    - diag_data
                    - torch.max(data_dash, dim=-1, keepdim=True).values
                )
                + eps
            )
        else:
            data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps
            )
        # print('data_dash2', data_dash.size())
        return data_dash.type_as(data)

    def linear_attention(self, q, k, v):
        k_cumsum = k.sum(dim=-2)
        D_inv = 1.0 / torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q))
        context = torch.einsum("...nd,...ne->...de", k, v)
        out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)
        return out

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)  # [1024, 50, 64]
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [1024, 2, 50, 32]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        create_kernel_q = partial(
            self.softmax_kernel,
            projection_matrix=self.projection_matrix_q,
            device=query_layer.device,
        )
        create_kernel_k = partial(
            self.softmax_kernel,
            projection_matrix=self.projection_matrix_k,
            device=query_layer.device,
        )
        query_layer = create_kernel_q(query_layer, is_query=True)
        key_layer = create_kernel_k(key_layer, is_query=False)
        # print('query_layer', query_layer.size())
        context_layer = self.linear_attention(query_layer, key_layer, value_layer)

        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()  # [1024, 50, 2, 32]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )  # [1024, 50, 64]
        context_layer = context_layer.view(*new_context_layer_shape)  # [1024, 50, 64]

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


### Common Light Layer and Encoder ###
class LightAttentionTransformerLayer(nn.Module):
    def __init__(
        self,
        config,
        n_heads,
        seq_len,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(LightAttentionTransformerLayer, self).__init__()
        model_name = config["model"]
        if model_name == "Linformer":
            self.multi_head_attention = LinformerAttention(
                config,
                n_heads,
                seq_len,
                hidden_size,
                hidden_dropout_prob,
                attn_dropout_prob,
                layer_norm_eps,
            )
        elif model_name == "Performer":
            self.multi_head_attention = PerformerAttention(
                config,
                n_heads,
                seq_len,
                hidden_size,
                hidden_dropout_prob,
                attn_dropout_prob,
                layer_norm_eps,
            )
        elif model_name == "Synthesizer":
            self.multi_head_attention = SynthesizerAttention(
                n_heads,
                seq_len,
                hidden_size,
                hidden_dropout_prob,
                attn_dropout_prob,
                layer_norm_eps,
            )
        elif model_name == "LinearTrm":
            self.multi_head_attention = LinearTrmAttention(
                n_heads,
                seq_len,
                hidden_size,
                hidden_dropout_prob,
                attn_dropout_prob,
                layer_norm_eps,
            )
        else:
            self.multi_head_attention = MultiHeadAttention(
                n_heads,
                seq_len,
                hidden_size,
                hidden_dropout_prob,
                attn_dropout_prob,
                layer_norm_eps,
            )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LightAttentionTransformerEncoder(nn.Module):
    def __init__(
        self,
        config,
        n_layers=2,
        n_heads=2,
        seq_len=50,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):

        super(LightAttentionTransformerEncoder, self).__init__()

        layer = LightAttentionTransformerLayer(
            config,
            n_heads,
            seq_len,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self, hidden_states, attention_mask=None, output_all_encoded_layers=True
    ):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


### LightSANs ###
class ItemToInterestAggregation(nn.Module):
    def __init__(self, seq_len, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests  # k latent interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):  # [B, L, d] -> [B, k, d]
        D_matrix = torch.matmul(input_tensor, self.theta)  # [B, L, k]
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        result = torch.einsum("nij, nik -> nkj", input_tensor, D_matrix)  # #[B, k, d]

        return result


class LightSANsAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LightSANsAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialization for low-rank decomposed self-attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attpooling_key = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )
        self.attpooling_value = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )

        # initialization for decoupled position encoding
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = (
            float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        )
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):  # transfor to multihead
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, pos_emb):
        # linear map
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # low-rank decomposed self-attention: relation of items
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.attpooling_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(
            self.attpooling_value(mixed_value_layer)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)

        # decoupled position encoding: relation of positions
        value_layer_pos = self.transpose_for_scores(mixed_value_layer)
        pos_emb = self.pos_ln(pos_emb).unsqueeze(0)
        pos_query_layer = (
            self.transpose_for_scores(self.pos_q_linear(pos_emb)) * self.pos_scaling
        )
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb))

        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        abs_pos_bias = nn.Softmax(dim=-2)(abs_pos_bias)

        context_layer_pos = torch.matmul(abs_pos_bias, value_layer_pos)

        context_layer = context_layer_item + context_layer_pos

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LightSANsLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(LightSANsLayer, self).__init__()
        self.multi_head_attention = LightSANsAttention(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, pos_emb):
        attention_output = self.multi_head_attention(hidden_states, pos_emb)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LightSANsEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        k_interests=5,
        hidden_size=64,
        seq_len=50,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):

        super(LightSANsEncoder, self).__init__()
        layer = LightSANsLayer(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, pos_emb, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, pos_emb)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


### HaloNet ###
class HaloNetEncoder(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_heads=2,
                 hidden_size=64,
                 seq_len=50,
                 inner_size=256,
                 block_size=10,
                 halo_size=6,
                 attn_scale_factor=2,
                 hidden_dropout_prob=0.5,
                 attn_dropout_prob=0.5,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12):
        super(HaloNetEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        layer = HaloNetLayer(n_heads, hidden_size, seq_len, inner_size, block_size, halo_size, attn_scale_factor,
                             hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(n_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class HaloNetLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, seq_len, intermediate_size, block_size, halo_size, attn_scale_factor,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(HaloNetLayer, self).__init__()
        self.multi_head_attention = HaloAttention(n_heads, hidden_size, seq_len, block_size, halo_size,
                                                  attn_scale_factor, hidden_dropout_prob,attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size,
                                         hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states):
        attention_output = self.multi_head_attention(hidden_states)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class HaloAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, seq_len, block_size, halo_size, attn_scale_factor, hidden_dropout_prob, attn_dropout_prob,
                layer_norm_eps):
        super(HaloAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.blocks = seq_len // block_size
        self.block_size = block_size
        self.halo_size = halo_size

        # initialization for low-rank decomposed self-attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # initialization for decoupled position encoding
        self.attn_scale_factor = attn_scale_factor
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):  # transfor to multihead
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        for i in range(self.blocks):
            mixed_query_layer = self.query(input_tensor[:, i*self.block_size:(i+1)*self.block_size, :])
            if i > 0:
                left = i*self.block_size-self.halo_size
            else:
                left = 0
            if i < self.blocks-1:
                right = (i + 1) * self.block_size+self.halo_size
            else:
                right = (i + 1) * self.block_size
            mixed_key_layer = self.key(input_tensor[:, left:right, :])
            mixed_value_layer = self.value(input_tensor[:, left:right, :])  # [batch_size, seq_len, hidden_size]

            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)

            if i == 0:
                all_context_layer = context_layer
            else:
                all_context_layer = torch.cat((all_context_layer, context_layer), dim=-2)

        all_context_layer = all_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = all_context_layer.size()[:-2] + (self.all_head_size,)
        all_context_layer = all_context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(all_context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # [batch_size, max_seq_len, hidden_size]

        return hidden_states
