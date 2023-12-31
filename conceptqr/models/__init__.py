from lightchain import Object 
import torch
from conceptqr.util import batch_iter, concatenate
from tqdm import tqdm 

class LM(Object):
    name = 'Language Model'
    def __init__(self, model, tokenizer, generation_kwargs={}, tokenizer_kwargs={}, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.batch_size = batch_size

    def generate(self, inp):
        prompt = self.tokenizer(inp, return_tensors="pt", **self.tokenizer_kwargs).to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(**prompt, **self.generation_kwargs)
        return self.tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)
    
    def __call__(self, inp):
        if len(inp) > self.batch_size:
            return concatenate([self.generate(prompt) for prompt in tqdm(batch_iter(inp, self.batch_size), desc="Generating...")])
        else:
            return self.generate(inp)
                
        
