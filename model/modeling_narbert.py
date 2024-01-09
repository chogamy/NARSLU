import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import math
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = int(args.d_model / args.n_heads)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        scores.masked_fill_(attn_mask, -1e9)
        last_attention_weight = scores

        attn = self.dropout(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)

        return context, attn, last_attention_weight


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.args = args
        self.d_k = int(args.d_model / args.n_heads)
        self.d_v = int(args.d_model / args.n_heads)
        self.n_heads = args.n_heads
        self.W_Q = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_K = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_V = nn.Linear(args.d_model, self.d_v * args.n_heads)
        self.li1 = nn.Linear(args.n_heads * self.d_v, args.d_model)
        self.layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)

        context, attn, last_attention_weight = ScaledDotProductAttention(self.args)(
            q_s, k_s, v_s, attn_mask
        )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )
        output = self.li1(context)

        return output + residual, attn, last_attention_weight


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        output = self.dropout(self.relu(self.conv1(inputs.transpose(1, 2))))
        output = self.conv2(output).transpose(1, 2)

        return output + residual


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.args = args
        self.dropout = args.dropout
        self.enc_self_attn = Attention(
            self.args.d_model, self.args.n_heads, self.args.dropout
        )
        self.pos_ffn = PoswiseFeedForwardNet(self.args)
        self.self_attn_layer_norm = nn.LayerNorm(args.d_model)
        self.final_layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, enc_inputs, enc_self_attn_mask=None):
        residual = enc_inputs
        hidden_states, attn, _ = self.enc_self_attn(
            hidden_states=enc_inputs,
            attention_mask=enc_self_attn_mask,
            layer_head_mask=None,
            output_attentions=True,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        enc_outputs = self.pos_ffn(hidden_states)
        enc_outputs = self.final_layer_norm(enc_outputs)

        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = Attention(
            d_model,
            n_heads,
            dropout,
            is_decoder=True,
        )
        self.dec_enc_attn = Attention(
            d_model,
            n_heads,
            dropout,
            is_decoder=True,
        )

        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.dropout = dropout

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        dec_inputs,
        encoder_hidden_states,
        dec_self_attn_mask,
        encoder_attention_mask,
    ):
        residual = dec_inputs
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.dec_self_attn(
            hidden_states=dec_inputs,
            attention_mask=dec_self_attn_mask,
            output_attentions=True,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            # cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.dec_enc_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=True,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        dec_outputs = self.pos_ffn(hidden_states)
        dec_outputs = self.final_layer_norm(dec_outputs)

        att_weights = (self_attn_weights, cross_attn_weights)

        return dec_outputs, att_weights


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, pad_id, num_layers, max_len, n_heads, d_ff, dropout
    ):
        super(Decoder, self).__init__()

        # self.args = args

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.src_emb = nn.Embedding(vocab_size, d_model, pad_id)
        # embedding for pad id is fixed

        self.pos_embedding = PositionalEncoding(d_model, max_len)

        self.projection = nn.Linear(d_model, vocab_size)
        self.pad_ids = pad_id

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        return expanded_attn_mask

    def forward(
        self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask
    ):
        input_embeds = self.src_emb(input_ids) + self.pos_embedding(input_ids)
        attention_mask = _expand_mask(attention_mask, input_embeds.dtype)

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, input_embeds.dtype, tgt_len=input_ids.size()[-1]
            )

        hidden_states = input_embeds

        for layer in self.layers:
            hidden_states, last_attention_weight = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                encoder_attention_mask,
            )

        logits = self.projection(hidden_states)

        return logits


class Encoder(nn.Module):
    def __init__(
        self, args, vocab_size, pad_ids, embed_tokens: Optional[nn.Embedding] = None
    ):
        super(Encoder, self).__init__()

        self.args = args
        self.pad_ids = pad_ids
        self.d_model = args.d_model

        if embed_tokens is not None:
            self.src_emb = embed_tokens
        else:
            self.src_emb = nn.Embedding(vocab_size, self.d_model, self.pad_ids)
        self.pos_embedding = PositionalEncoding(self.d_model, args.max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(self.args) for _ in range(self.args.n_layers)]
        )

        # self.dropout = nn.Dropout(args.dropout)
        # self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, enc_inputs, attention_mask=None):
        enc_outputs = self.src_emb(enc_inputs) + self.pos_embedding(enc_inputs)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, enc_outputs.dtype)

        for layer in self.layers:
            enc_outputs, _ = layer(enc_outputs, attention_mask)

        return enc_outputs


class NARBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(NARBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        bslot_label_lst = [token for token in slot_label_lst if "I-" not in token]
        islot_label_lst = [token for token in slot_label_lst if "B-" not in token]

        self.num_bslot_labels = len(bslot_label_lst)
        self.num_islot_labels = len(islot_label_lst)
        self.mask_id = slot_label_lst.index("MASK")

        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = None
        if self.args.task in ["atis", "snips"]:
            self.intent_classifier = IntentClassifier(
                config.hidden_size, self.num_intent_labels, args.dropout_rate
            )
        # self.slot_classifier = SlotClassifier(
        #     config.hidden_size, self.num_slot_labels, args.dropout_rate
        # )

        self.slot_classifier = None
        self.bslot_classifier = None
        self.islot_classifier = None
        if args.n_dec == 1:
            self.slot_classifier = Decoder(
                vocab_size=self.num_slot_labels,
                d_model=config.hidden_size,
                pad_id=slot_label_lst.index("PAD"),
                num_layers=1,
                max_len=args.max_seq_len,
                # n_heads=8,
                n_heads=1,
                # d_ff=config.hidden_size * 4,
                d_ff=config.hidden_size,
                dropout=args.dropout_rate,
            )

        if args.n_dec == 2:
            self.bslot_classifier = Decoder(
                vocab_size=self.num_bslot_labels,
                d_model=config.hidden_size,
                pad_id=slot_label_lst.index("PAD"),
                num_layers=1,
                max_len=args.max_seq_len,
                n_heads=8,
                d_ff=config.hidden_size * 4,
                dropout=args.dropout_rate,
            )

            self.islot_classifier = Decoder(
                vocab_size=self.num_islot_labels,
                d_model=config.hidden_size,
                pad_id=slot_label_lst.index("PAD"),
                num_layers=1,
                max_len=args.max_seq_len,
                n_heads=8,
                d_ff=config.hidden_size * 4,
                dropout=args.dropout_rate,
            )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        intent_label_ids,
        slot_labels_ids,
        bslot_labels_ids,
        islot_labels_ids,
    ):
        model_output = {}
        model_output["slot_labels_ids"] = slot_labels_ids

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        model_output["intent_logits"] = intent_logits

        dec_input_ids = torch.full(
            input_ids.size(),
            self.mask_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        if self.args.n_dec == 1:
            slot_logits = self.slot_classifier(
                input_ids=dec_input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=sequence_output,
                encoder_attention_mask=attention_mask,
            )
            model_output["slot_logits"] = slot_logits
            model_output["bslot_logits"] = None
            model_output["islot_logits"] = None
        elif self.args.n_dec == 2:
            bslot_logits = self.bslot_classifier(
                input_ids=dec_input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=sequence_output,
                encoder_attention_mask=attention_mask,
            )

            islot_logits = self.islot_classifier(
                input_ids=dec_input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=sequence_output,
                encoder_attention_mask=attention_mask,
            )
            model_output["slot_logits"] = None
            model_output["bslot_logits"] = bslot_logits
            model_output["islot_logits"] = islot_logits

        model_output["loss"] = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1), intent_label_ids.view(-1)
                )
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels),
                    intent_label_ids.view(-1),
                )
            model_output["loss"] += intent_loss

        # 2. Slot Softmax
        # if slot_labels_ids is not None and self.slot_classifier is not None:
        if self.args.n_dec == 1:
            if self.args.use_crf:
                slot_loss = self.crf(
                    slot_logits,
                    slot_labels_ids,
                    mask=attention_mask.byte(),
                    reduction="mean",
                )
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    # print(slot_logits.shape)
                    # print(attention_mask.shape)
                    # assert 1 == 0
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[
                        active_loss
                    ]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(
                        slot_logits.view(-1, self.num_slot_labels),
                        slot_labels_ids.view(-1),
                    )
            model_output["loss"] += slot_loss
            # total_loss += self.args.slot_loss_coef * slot_loss
            # outputs = ((intent_logits, slot_logits),) + outputs[
            #     2:
            # ]  # add hidden states and attention if they are here

            # outputs = (total_loss,) + outputs

            # return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
        else:
            assert bslot_labels_ids is not None
            assert islot_labels_ids is not None

            bslot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)

            bslot_loss = bslot_loss_fct(
                bslot_logits.view(-1, self.num_bslot_labels),
                bslot_labels_ids.view(-1),
            )

            islot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            islot_loss = islot_loss_fct(
                islot_logits.view(-1, self.num_islot_labels),
                islot_labels_ids.view(-1),
            )

            slot_loss = bslot_loss + islot_loss

            model_output["loss"] += slot_loss

        return model_output

            # outputs = ((intent_logits, bslot_logits, islot_logits),) + outputs[
            #     2:
            # ]  # add hidden states and attention if they are here

            # outputs = (total_loss,) + outputs

            # return outputs
