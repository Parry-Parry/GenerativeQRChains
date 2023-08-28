from lightchain import Chain
from sentence_transformers import util

class Prune(Chain):
    def __init__(self, model, max_terms : int = 20, out_attr : str = 'expansion_terms'):
        super(Prune, self).__init__(model=model, name="Prune")
        self.max_terms = max_terms
        self.out_attr = out_attr

    def compute_similarity(self, query, tokens):
        tokens = tokens.split(' ')
        query_emb = self.model.encode(query, convert_to_tensor=True)
        tok_emb = self.model.encode(tokens, convert_to_tensor=True)
        cos_sim = util.cos_sim(query_emb, tok_emb).squeeze()
        sorted_idx = cos_sim.argsort(descending=True)
        new_tokens = ' '.join([tokens[i] for i in sorted_idx[:self.max_terms]])
        return new_tokens

    def logic(self, inp):
        out = inp.copy()
        out[self.out_attr] = out.apply(lambda x : self.compute_similarity(x['query'], x['expansion_terms']), axis=1)
        return out