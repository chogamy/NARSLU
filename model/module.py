import torch
from torch import nn
import torch.nn.functional as F


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.0):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


def exists(val):
    return val is not None


# Non-autoregressive training logic


def eojeol_mask(x, space_id, mask_id, pad_id):
    inp = torch.where((x == space_id) | (x == -100), x, mask_id).long()  # .to(x.device)
    inp = torch.where(inp == -100, pad_id, inp)
    return inp


def uniform_mask(x, mask_id, pad_id):
    b, l = x.size()

    inp = x.clone()

    lengths = torch.count_nonzero(x != -100, dim=-1)
    for i in range(b):
        num_to_mask = torch.randint(0, lengths[i], (1, 1))
        where_to_mask = torch.randperm(lengths[i].tolist())[:num_to_mask]
        inp[i][where_to_mask] = mask_id

    inp = torch.where(inp == -100, pad_id, inp)

    return inp


def random_mask(x, mask_id):
    to_mask = torch.randint(0, 2, x.size())
    inp = torch.where(to_mask == 0, mask_id, x)
    return inp


def full_mask(x, mask_id, pad_id):
    inp = torch.where(x == -100, pad_id, mask_id).long()  # .to(x.device)
    return inp


# Non autoregressive generation


# Select low-confidence tokens for masking
def select_mask(out, ratio, pad_id):
    b, l = out["sequence"].shape

    lengths = torch.count_nonzero(out["sequence"] != pad_id, dim=-1)
    num_to_mask = (lengths * ratio).int()

    indices = [torch.topk(-out["probs"][i], num_to_mask[i])[1] for i in range(b)]

    for i in range(b):
        batch = torch.zeros_like(indices[i])
        batch[:] = i
        print(batch)
        print(indices[i])
        print(torch.cat(batch, indices[i]))
        print(torch.stack(batch, indices[i]))

    assert 1 == 0

    mask_indices = None
    return mask_indices


# Mask-Predict
def select_worst(token_probs, num_mask):
    bsz, seq_len = token_probs.size()
    masks = [
        token_probs[batch, :].topk(
            max(1, num_mask[batch]), largest=False, sorted=False
        )[1]
        for batch in range(bsz)
    ]
    masks = [
        torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0)
        for mask in masks
    ]
    return torch.stack(masks, dim=0)


def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b * l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y


# Non-autoregressive wrapper class


# dec_training_strategy = {uniform, full_mask, ctc, information}


class NonAutoregressiveWrapper(nn.Module):
    def __init__(self, net, mask_index, *, pad_value=0, mask_prob=0.0, **kwargs):
        super().__init__()
        self.mask_index = mask_index
        self.pad_value = pad_value
        self.ignore_index = -100

        self.net = net
        self.max_seq_len = kwargs["max_seq_len"]
        self.train_strategy = kwargs.pop("dec_train_strategy", None)
        if self.train_strategy == "information":
            # 여기를 좀 general, special
            self.space_id = kwargs.pop("space_id", None)

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.0
        self.mask_prob = mask_prob

    @torch.no_grad()
    def generate(self, start_tokens, lengths, iteration=1, **kwargs):
        out = {"sequence": start_tokens}

        """
        lp_out = {
                'dec_inp',  # 토큰일때 size, ctc일때 size 다름
                'lengths',  # 사이즈 정의
                'probs'     # 정의
            }
        """
        if len(start_tokens.shape) == 2:
            start_tokens[start_tokens == 1] = self.mask_index

        mask = out["sequence"] != self.pad_value
        kwargs.update(self_attn_context_mask=mask)

        was_training = self.net.training

        b = start_tokens.size(0)

        self.net.eval()

        # iterative refinement
        total_iteration = iteration
        while iteration > 0:
            logits = self.net(out["sequence"], **kwargs)

            # logits = logits / 5

            # logits[:, :, 0] = -1e10
            # logits[:, :, 1] = -1e10
            # logits[:, :, 2] = -1e10

            T = 1
            logits = logits / T

            probs = F.softmax(logits, dim=-1)
            token_probs, tokens = probs.max(dim=-1)

            b, _ = tokens.size()

            for i in range(b):
                tokens[i][lengths[0][i] :] = self.pad_value
                token_probs[i][lengths[0][i] :] = 1.0

            """
            여기 한번 padding 안되나?
            """
            out["logits"] = logits
            out["sequence"] = tokens
            out["probs"] = token_probs
            out["scores"] = probs

            iteration -= 1
            if iteration == 0:
                break

            # Mask-Predict code
            # lengths = torch.count_nonzero(out['sequence']!=self.pad_value, dim=-1)
            # num_to_mask = (lengths * (iteration / total_iteration)).long()
            # mask = select_worst(out['probs'], num_to_mask)
            # assign_single_value_long(out['sequence'], mask, self.mask_index)

            # WIP
            # mask_indices = select_mask(out, iteration / total_iteration, self.pad_value)
            # out['sequence'][mask_indices] = self.mask_index

        # if num_dims == 1:
        #     out['sequence'] = out['sequence'].squeeze(0)

        self.net.train(was_training)

        return out

    def forward(self, x, **kwargs):
        """
        kwargs = {context, context_mask}
        """
        inp = torch.full(x.shape, self.mask_index, dtype=torch.long, device=x.device)

        # inp = full_mask(x, self.mask_index, self.pad_value)
        kwargs.pop("inp", None)

        if len(inp.shape) == 2:
            mask = inp != self.pad_value
            kwargs.update(self_attn_context_mask=mask)

        out = self.net(inp, **kwargs)

        return out

        # loss = F.cross_entropy(out.transpose(1, 2), x)

        # return loss
