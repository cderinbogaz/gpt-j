import time
import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2Tokenizer
import torch
import transformers
import collections
import os
import json

class Checkpoint(collections.MutableMapping):
    def __init__(self):
        self.checkpoint = torch.load("pytorch_model.bin")
        print("Loaded")
    def __len__(self):
        return len(self.checkpoint)
    def __getitem__(self, key):
        return torch.load(self.checkpoint[key])
    def __setitem__(self, key, value):
        return
    def __delitem__(self, key, value):
        return
    def keys(self):
        return self.checkpoint.keys()
    def __iter__(self):
        for key in self.checkpoint:
            yield (key, self.__getitem__(key))
    def __copy__(self):
        return self.__dict__
    def copy(self):
        return self.__dict__

print("load", flush=True)
with open('config.json', 'r') as f:
    config = json.load(f)

model = GPTNeoForCausalLM.from_pretrained(pretrained_model_name_or_path=None, config=config, state_dict=Checkpoint())
print("ok")
model.eval()

model = GPTNeoForCausalLM.from_pretrained("./gpt-j-hf")
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model.half().cuda()


input_text = "J K Rowling Author Biography:"
input_ids = tokenizer.encode(str(input_text), return_tensors='pt').cuda()

output = model.generate(
    input_ids,
    do_sample=True,
    max_length=20,
    top_p=0.7,
    top_k=0,
    temperature=1.0,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
