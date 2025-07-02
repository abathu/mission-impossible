# perplexities.py
# Author: Julie Kallini

# For importing utils
import sys
# sys.path.append("..")
sys.path.append("/home/s2678328/mission-impossible_fork")

from transformers import GPT2LMHeadModel
from gpt2_no_positional_encoding_model import GPT2NoPositionalEncodingLMHeadModel
from utils import CHECKPOINT_READ_PATH, PERTURBATIONS, BABYLM_DATA_PATH, \
    PAREN_MODELS #, gpt2_original_tokenizer
from tqdm import tqdm
from glob import glob
from numpy.random import default_rng
import pandas as pd
import torch
import itertools
import argparse
import os
import tokenizers
from transformers import AutoTokenizer


MAX_TRAINING_STEPS = 3000
CHECKPOINTS = list(range(100, MAX_TRAINING_STEPS+1, 100))

gpt2_original_tokenizer  =AutoTokenizer.from_pretrained("/home/s2678328/tokenizers/tokenzier_Russain/auto_tokenizer_auto",use_fast=True,)


def create_attention_mask(token_lists):
    seq_length = max([len(i) for i in token_lists])
    batch_size = len(token_lists)
    mask = torch.full((batch_size, seq_length), 0)

    for i, tokens in enumerate(token_lists):
        mask[i, 0:len(tokens)] = 1

    return mask


def create_input_ids(token_lists, pad_token_id):
    padded = zip(*itertools.zip_longest(*token_lists, fillvalue=pad_token_id))
    return torch.tensor(list(padded))


def get_perplexities(model, token_lists, pad_token_id, device="cuda"):

    # Prepare data
    input_ids = create_input_ids(token_lists, pad_token_id).to(device)
    labels = input_ids.clone()  # GPT-2 uses input as labels for CLM task
    attention_mask = create_attention_mask(token_lists).to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask)

    # The "shifted" nature of labels in GPT-2 (next token prediction)
    # Shift logits, labels, and attention mask by one position
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # Instantiate loss function with no reduction
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    # Calculate per-token loss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    # Reshape back to the original batch size and sequence length
    loss = loss.view(shift_labels.size())

    # Apply the attention mask - only calculate loss where mask is 1
    loss = loss * shift_attention_mask

    # Sum the loss over the sequence length, get per-example perplexity
    per_example_loss = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
    return torch.exp(per_example_loss).tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Edge probing',
        description='Edge probing experiments')
    
    