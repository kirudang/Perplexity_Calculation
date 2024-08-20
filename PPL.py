import os
# Set the HF_HOME environment variable to point to the desired cache location
os.environ["HF_TOKEN"] = "your_HF_token"
# Specify the directory path 
cache_dir = 'your_cache_dirictory'
# Set the HF_HOME environment variable
os.environ['HF_HOME'] = cache_dir

import csv
import sys, argparse, time, json
from accelerate import Accelerator
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch

def get_perplexities(model, tokenizer, text, device):
    """Returns the perplexities of the model for the given texts.
    
    Args:
        model: The model which outputs logits for the next token.
        tokenizer: The tokenizer to encode the text.
        text: The input text to compute perplexities for.
        device: The device to run the model on.
    
    Returns:
        The perplexities of the model for the given texts as a tensor of shape (B,).
    """
    model = model.to(device)
    ids = tokenizer.encode(text, return_tensors='pt').to(device)
    B, T = ids.shape
    
    perplexities = torch.zeros(B).to(ids.device)
    
    with torch.no_grad():
        for i in range(T-1):
            l_t = model(ids[:, :i+1])[:, -1, :]
            l_t = torch.softmax(l_t, dim=-1)
            l_t = l_t[range(B), ids[:, i+1]]
            l_t = torch.log(l_t)
            perplexities += l_t
    
    return torch.exp(-perplexities / (T-1))

def main(args):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Device
    device = Accelerator().device
    print(device)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.tokenizer, cache_dir=cache_dir)
    model = model.to(device)

    # Define Model wrapper to get the logit
    class Model_Wrapper(torch.nn.Module):
        """A wrapper around the model to take ids as input and return logits as output."""

        def __init__(self, model, tokenizer, device):
            super(Model_Wrapper, self).__init__()
            self.tokenizer = tokenizer
            self.model = model
            self.device = device

        def forward(self, input_ids):
            outputs = self.model(input_ids)
            return outputs.logits

    # Create an instance of Model_Wrapper
    model = Model_Wrapper(model, tokenizer, device).to(device)   

    # Setting seed
    set_seed(0)

    # Load output text from json file
    with open(args.data, 'r') as f:
        data = json.load(f)

    output_text = [item[args.type_of_data] for item in data if args.type_of_data in item]

    # Loop through the output text and detect the watermark
    results = []
    for i, item in enumerate(output_text):
        num_tokens = len(item.split())
        #print(f"Item number: {i}")
        
        if num_tokens >= 16: # Only consider items with at least 16 tokens for valid assessment
            perplexities = get_perplexities(model,tokenizer, item, device = device).item()
            print(f' PPL of item {i} is: {perplexities}')
            results.append([i, perplexities])

        else:
            print(f"Skipping item number {i} due to insufficient tokens.")
    # Write the results to a CSV file
    with open('llama2_PPL.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["data_item", 'perplexities'])  # Write the header
        writer.writerows(results)  # Write the data



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate PPL')
    parser.add_argument('--data',default= 'llama2.json',type=str, help='a file containing the document to test')
    parser.add_argument('--type_of_data',default='text',type=str, help='column from data file to use as input to the watermark test')
    parser.add_argument('--tokenizer',default='meta-llama/Llama-2-7b-chat-hf',type=str, help='Tokenizer and model for the PPL calculation')

    main(parser.parse_args())