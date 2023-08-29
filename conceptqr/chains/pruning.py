from lightchain import Chain
import numpy as np
from sentence_transformers import util

class Prune(Chain):
    def __init__(self, model, max_terms : int = 20, out_attr : str = 'expansion_terms'):
        super(Prune, self).__init__(model=model, name="Prune")
        self.max_terms = max_terms
        self.out_attr = out_attr

    def compute_similarity(self, query, tokens):
        tokens = [t for t in tokens.split(' ') if len(t) > 1]
        query_emb = self.model.encode(query, convert_to_tensor=True)
        tok_emb = self.model.encode(tokens, convert_to_tensor=True)
        cos_sim = util.cos_sim(query_emb, tok_emb).squeeze()
        sorted_idx = cos_sim.argsort(descending=True)
        new_tokens = ' '.join([tokens[i] for i in sorted_idx[:self.max_terms]])
        return new_tokens

    def logic(self, inp):
        out = inp.copy()
        out[self.out_attr] = out.apply(lambda x : self.compute_similarity(x['query'], x[self.out_attr]), axis=1)
        return out

class IDFPrune(Chain):
    b = 0.5
    def __init__(self, index_path : str, stemmer : str = 'PorterStemmer', topk : int = 20, out_attr='expansion_terms'):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        super().__init__()
        self.index = pt.IndexFactory.of(pt.get_dataset(index_path).get_index('terrier_stemmed'), memory=True)
        self.lexicon = self.index.getLexicon()
        self.collection_stats = self.index.getCollectionStatistics()
        self.num_docs = self.collection_stats.getNumberOfDocuments()

        stem_name = f"org.terrier.terms.{stemmer}" if '.' not in stemmer else stemmer
        self.stemmer = pt.autoclass(stem_name)().stem
        self.stopwords = pt.autoclass("org.terrier.terms.Stopwords")(None).isStopword
        self.topk = topk
        self.out_attr = out_attr
    
    def idf(self, token):
        if self.stopwords(token): return 0.
        try:
            term_entry = self.lexicon[self.stemmer(token)]
        except KeyError:
            print(f"Token {token} not found in index")
            return 0.
        
        df = term_entry.getDocumentFrequency()
        idf = np.log((self.num_docs-df+self.b) / (df+self.b))

        return idf
    
    def get_topk(self, tokens):
        scored = [(token, self.idf(token)) for token in tokens.split(' ') if len(token) > 1]
        scored = sorted(scored, key=lambda x : x[1], reverse=True)
        return ' '.join([x[0] for x in scored[:self.topk]])

    def logic(self, inp):
        out = inp.copy()
        out[self.out_attr] = out[self.out_attr].apply(lambda x : self.get_topk(x))
        return out

