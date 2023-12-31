import pyterrier as pt
if not pt.started():
    pt.init()
from lightchain import LambdaChain
from fire import Fire
from conceptqr.chains.weighting import CWPRF_Weighting
from conceptqr.util import concatenate
import torch
import pandas as pd
from typing import List
import ast

def concatenate_concepts(inp):
    # group by qid and concatenate expansion_terms over concept columns
    inp = inp.groupby(['qid', 'query', 'concept'])['expansion_terms'].agg(list).reset_index()
    inp = inp.groupby(['qid', 'query'])['expansion_terms'].apply(lambda x: [term for terms in x for term in terms]).reset_index()
    inp['expansion_terms'] = inp['expansion_terms'].apply(lambda x : ' '.join(concatenate(x)))
    return inp

def main(weight_name_or_path : str, 
         intermediate : str,
         out_path : str,
         stopwords : str = None,
         beta : float = 0.5,
         topk : int = 20,):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cwprf = CWPRF_Weighting(weight_name_or_path, 
                            topk=topk, 
                            beta=beta, 
                            stopwords=True if stopwords else False, 
                            stopword_path = stopwords, 
                            weight_mode=None, 
                            id_mode='mean', 
                            device=device)
    
    ConceptConcatenation = LambdaChain(concatenate_concepts)
    
    pipe = ConceptConcatenation >> cwprf

    topics = pd.read_csv(intermediate, sep='\t', index_col=False, converters={'expansion_terms': ast.literal_eval})

    new_queries = pipe(topics)
    new_queries.to_csv(out_path, sep='\t', index=False)

if __name__ == "__main__":
    Fire(main)