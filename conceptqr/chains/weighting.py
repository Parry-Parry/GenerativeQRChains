import pyterrier as pt
if not pt.started():
    pt.init()

from pyterrier.model import push_queries

from lightchain import Object 
from abc import abstractmethod
from transformers import BertTokenizer
import os
import subprocess as sp
import pandas as pd
import urllib
import torch
import numpy as np

class WeightingModel(Object):
    essential = ['qid', 'query', 'expansion_terms']
    name = 'WeightingModel'
    def __init__(self):
        pass

    @abstractmethod
    def logic(self, inp):
        raise NotImplementedError
    
    def __call__(self, inp):
        for col in self.essential:
            assert col in inp.columns, f"WeightingModel requires '{col}' in input"
        out = inp.copy()
        out = push_queries(out, keep_original=True)
        out['query'] = self.logic(out[self.essential])
        out = out.drop('expansion_terms', axis=1)
        return out

class CWPRF_Weighting(WeightingModel):
    max_length = 512
    special_ids = [101,102,1,2]
    def __init__(self, 
                 model_name_or_path : str, 
                 topk : int = 20,
                 beta : float = 0.5,
                 stopwords : bool = False,
                 weight_mode = None,
                 id_mode = 'mean',
                 stopword_path = '/stopword-list.txt',
                 batch_size = 32,
                 device = None,
                 ):
        super().__init__()
        from conceptqr.models.cwprf import CWPRFEncoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = CWPRFEncoder.from_pretrained("castorini/unicoil-msmarco-passage")
        self.tokenizer = BertTokenizer.from_pretrained("castorini/unicoil-msmarco-passage")
        checkpoint = torch.load(model_name_or_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        self.topk = topk
        self.beta = beta
        self.stopwords = stopwords
        self.weight_mode = weight_mode
        self.id_mode = id_mode
        self.batch_size = batch_size # Not currently used

        self.stoplist = self.init_stopwords(stopword_path) if stopwords else None
    
    def init_stopwords(self, path):
        if not os.path.isfile(path):
            urllib.request.urlretrieve("https://raw.githubusercontent.com/terrier-org/terrier-core/5.x/modules/core/src/main/resources/stopword-list.txt", path)
            sp.run('wget  -O  {path}', shell=True)
        with open(path) as f:
            words = list(map(lambda x : x.strip(), f.readlines()))
        return [x for x in self.tokenizer.convert_tokens_to_ids(words) if x != 100]
    
    def pivot(self, text):
        frame = []
        for i, token in enumerate(text.split(' ')):
            tokenised = self.tokenizer(token, return_tensors='pt')
            for j in range(tokenised.input_ids.shape[1]):
                frame.append({'word' : token, 'pos' : i, 'id' : tokenised.input_ids[0, j].item()})
        return pd.DataFrame.from_records(frame)
    
    def assign_weights(self, token, weight):
        adjusted_weight = weight * self.beta
        return f'{token}^{adjusted_weight:.4f}'
    
    def compute_weights (self, expansion):
        idx = expansion['id'].tolist()
        expansion_ids = torch.tensor(idx)
        expansion_ids = expansion_ids.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=expansion_ids, attention_mask=torch.ones_like(expansion_ids)).cpu()
        return outputs.squeeze(dim=2).squeeze(dim=0).numpy().tolist()
    
    def filter(self, df):
        query_tokens = df[df['query']]['word'].unique().tolist()
        df = df[~df['word'].isin(query_tokens)]
        df = df[~df['id'].isin(self.special_ids)]
        if self.stopwords: df = df[~df['id'].isin(self.stoplist)]
        df = df[~df['query']] 
        return df.groupby(['word', 'pos', 'query']).agg({'output_weights' : self.id_mode}).reset_index()

    def weight_terms(self, query, expansion_terms):
        # Expand tokens
        query_pivot = self.pivot(query)

        query_pivot['query'] = True
        expansion_pivot = self.pivot(expansion_terms)
        expansion_pivot['query'] = False
        # Truncate
        if len(query_pivot) + len(expansion_pivot) > self.max_length:
            expansion_pivot = expansion_pivot.iloc[:self.max_length - len(query_pivot)]
        # Compute Weights
        expansion_tokens = pd.concat([query_pivot, expansion_pivot])
        expansion_tokens['output_weights'] = self.compute_weights(expansion_tokens)
        # Filter and Aggregate
        expansion_tokens = self.filter(expansion_tokens)

        potential_candidates = expansion_tokens[expansion_tokens['query'] == False]
        potential_candidates = potential_candidates.sort_values(by='output_weights', ascending=False)
        candidates = potential_candidates.drop_duplicates(subset='word').head(self.topk)
        
        return query + ' ' + ' '.join(candidates.apply(lambda x : self.assign_weights(x['word'], x['output_weights']), axis=1))

    def logic(self, inp):
        out = inp.copy()
        return out.apply(lambda x : self.weight_terms(x['query'], x['expansion_terms']), axis=1)

class FixedWeighting(WeightingModel):
    def __init__(self, topk : int = 20, beta : float = 0.5):
        super().__init__()
        self.topk = topk
        self.beta = beta
        self.stopwords = pt.autoclass("org.terrier.terms.Stopwords")(None).isStopword
    
    def join_terms(self, query, expansion_terms):
        query_terms = query.split(' ')
        terms = [term for term in expansion_terms.lower().split(' ') if not self.stopwords(term) and term not in query_terms and len(term) > 1]
        return query + ' ' + ' '.join(f'{term}^{self.beta:.4f}' for term in terms[:self.topk])
    
    def logic(self, inp):
        out = inp.copy()
        return out.apply(lambda x : self.join_terms(x['query'], x['expansion_terms']), axis=1)

class TFIDFWeighting(WeightingModel):
    # Uses probabilistic smoothed idf
    b = 0.5
    def __init__(self, index_path : str, stemmer : str = 'PorterStemmer', topk : int = 20):
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

    def tfidf(self, token):
        if self.stopwords(token): return 0.
        try:
            term_entry = self.lexicon[self.stemmer(token)]
        except KeyError:
            print(f"Token {token} not found in index")
            return 0.

        tf = term_entry.getFrequency()
        df = term_entry.getDocumentFrequency()
        idf = np.log((self.num_docs-df+self.b) / (df+self.b))

        return tf * idf
    
    def logic(self, inp):
        out = inp.copy()

        out['scores'] = out['expansion_terms'].apply(lambda x : [(term, self.tfidf(term)) for term in x.lower().split(' ') if len(term) > 1])
        out['scores'] = out['scores'].apply(lambda x : sorted(x, key=lambda x : x[1], reverse=True))
        out['scores'] = out['scores'].apply(lambda x : x[:self.topk])
        out['scores'] = out.apply(lambda x : [(term, score) for term, score in x['scores'] if score > 0. and term.lower() not in x['query'].split(' ')], axis=1)
        out['expansion_terms'] = out['scores'].apply(lambda x : ' '.join([f'{term}^{score:.4f}' for term, score in x]))

        return out.apply(lambda x : f"{x['query']} {x['expansion_terms']}", axis=1)



    

        
        