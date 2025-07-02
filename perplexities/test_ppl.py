import os
import torch
import itertools
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm


def create_attention_mask(token_lists):
    seq_length = max(len(t) for t in token_lists)
    batch_size = len(token_lists)
    mask = torch.zeros((batch_size, seq_length), dtype=torch.long)

    for i, tokens in enumerate(token_lists):
        mask[i, :len(tokens)] = 1

    return mask


def create_input_ids(token_lists, pad_token_id):
    padded = list(zip(*itertools.zip_longest(*token_lists, fillvalue=pad_token_id)))
    return torch.tensor(padded, dtype=torch.long)


def get_perplexities(model, token_lists, pad_token_id, device="cuda"):
    model.eval()

    input_ids = create_input_ids(token_lists, pad_token_id).to(device)
    labels = input_ids.clone()
    attention_mask = create_attention_mask(token_lists).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size()) * shift_attention_mask

    per_example_loss = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
    return torch.exp(per_example_loss).tolist()


if __name__ == "__main__":
    # === Path Setup ===
    tokenizer_path = "/home/s2678328/tokenizers/tokenzier_Russain/auto_tokenizer_auto"
    model_path = "/home/s2678328/nlp/llms-in-llms/babylm_models/babylm_shuffle_control_100M_randinit/babylm_shuffle_control_100M_randinit_seed41/runs/babylm_shuffle_control_100M_randinit_seed41/checkpoint-3000"

    # === Load tokenizer and model ===
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True).to("cuda")

    # === Sample text ===
    toks = "я его очень боюсь."
    test_sents = tokenizer([toks], add_special_tokens=True)["input_ids"]

    # === Calculate perplexity ===
    ppls = get_perplexities(model, [test_sents], tokenizer.eos_token_id)
    print("Perplexity:", ppls)