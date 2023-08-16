from lightchain import Object 
from ..util import batch_iter, concatenate

class FLANT5(Object):
    def __init__(self, model, tokenizer, generation_kwargs={}, tokenizer_kwargs={}, batch_size=8):
        super(FLANT5, self).__init__(name="FLANT5")
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.batch_size = batch_size

    def generate(self, inp):
        prompt = self.tokenizer(inp, return_tensors="pt", **self.tokenizer_kwargs)
        generated = self.model.generate(**prompt, **self.generation_kwargs)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    def __call__(self, inp):
        if len(inp) > self.batch_size:
            return concatenate([self.generate(prompt) for prompt in batch_iter(inp, self.batch_size)])
        else:
            return self.generate(inp)
                
        