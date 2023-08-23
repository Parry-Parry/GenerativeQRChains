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

import torch

class WeightingModel(Object):
    essential = ['qid', 'query', 'expansion_terms']
    def __init__(self):
        super(WeightingModel, self).__init__(name="Weighting_Model")

    @abstractmethod
    def logic(self, inp):
        raise NotImplementedError
    
    def __call__(self, inp):
        for col in self.essential:
            assert col in inp.columns, f"WeightingModel requires '{col}' in input"
        out = inp.copy()
        out['weighted_terms'] = self.logic(out[self.essential])
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
        self.model = CWPRFEncoder.from_pretrained(model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        
        self.topk = topk
        self.beta = beta
        self.stopwords = stopwords
        self.weight_mode = weight_mode
        self.id_mode = id_mode
        self.batch_size = batch_size # Not currently used

        self.stoplist = self.init_stopwords(stopword_path) if stopwords else None
    
    def init_stopwords(self, path):
        if not os.path.exists(path):
            sp.run('wget https://raw.githubusercontent.com/terrier-org/terrier-core/5.x/modules/core/src/main/resources/stopword-list.txt -O  {path}', shell=True)
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
        expansion_ids = torch.cat(expansion['id'].tolist()).unsqueeze(0)
        expansion_ids = expansion_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=expansion_ids, attention_mask=torch.ones_like(expansion_ids)).cpu()
        return outputs.squeeze(dim=2).squeeze(dim=0).numpy().tolist()
    
    def filter(self, df):
        df = df[~df['query']] 
        df = df[~df['id'].isin(self.special_ids)]
        if self.stopwords: df = df[~df['id'].isin(self.stoplist)]
        return df.groupby(['word', 'pos', 'query']).agg({'output_weight' : self.id_mode}).reset_index()

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
        expansion_tokens['output_weight'] = self.compute_weights(expansion_tokens)
        # Filter and Aggregate
        expansion_tokens = self.filter(expansion_tokens)

        potential_candidates = expansion_tokens[expansion_tokens['query'] == False]
        potential_candidates = potential_candidates.sort_values(by='output_weights', ascending=False)
        candidates = potential_candidates.drop_duplicates(subset='word').head(self.topk)
        
        return query + ' ' + ' '.join(candidates.apply(lambda x : self.assign_weights(x['word'], x['outputs_weights']), axis=1))

    def logic(self, inp):
        out = inp.copy()
        expansion = out.apply(lambda x : self.weight_terms(x['query'], x['expansion_terms']), axis=1)
        out = push_queries(out, keep_original=True)
        out['query'] = expansion
        return out
        
        