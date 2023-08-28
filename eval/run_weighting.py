import pyterrier as pt
if not pt.started():
    pt.init()
from fire import Fire
from conceptqr.chains.weighting import CWPRF_Weighting, TFIDFWeighting, FixedWeighting
from conceptqr.chains.pruning import Prune
import torch
import pandas as pd
import ast
from types import SimpleNamespace 
import ir_measures
from ir_measures import *
import ir_datasets as irds
import json
import os

def init_weighting(hparams : SimpleNamespace):
    if hparams.weighting == 'cwprf':
        weighting = CWPRF_Weighting(hparams.weight_name_or_path,
                                 topk=hparams.topk, 
                                 beta=hparams.beta, 
                                 stopwords=True if hparams.stopwords else False, 
                                 stopword_path = hparams.stopwords, 
                                 weight_mode=None, 
                                 id_mode='mean', 
                                 device=hparams.device)
    elif hparams.weighting == 'tfidf':
        weighting = TFIDFWeighting(hparams.index_path,
                              topk=hparams.topk,)
    elif hparams.weighting == 'fixed':
        weighting = FixedWeighting(hparams.topk, hparams.beta)
    
    if hparams.pruning:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-distilroberta-v1')
        prune = Prune(model, max_terms=hparams.pruning)

        weighting = prune >> weighting
    
    return weighting
                              
def main(intermediate : str,
         out_path : str,
         weight_name_or_path : str = None,
         stopwords : str = None,
         mode : str = 'fixed',
         beta : float = 0.5,
         topk : int = 20,
         prune_k : int = None,):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hparams = SimpleNamespace(
        weighting = mode,
        weight_name_or_path = weight_name_or_path,
        topk = topk,
        beta = beta,
        stopwords = stopwords,
        device = device,
        pruning = prune_k,
    )
    
    pipe = init_weighting(hparams)
    topics = pd.read_csv(intermediate, sep='\t', index_col=False)

    new_queries = pipe(topics)
    new_queries.to_csv(out_path, sep='\t', index=False)

    qrels = irds.load("msmarco-passage/trec-dl-2019/judged").qrels_iter()
    evaluator = ir_measures.evaluator([nDCG@10, R(rel=2)@100, R(rel=2)@1000, P(rel=2)@10, P(rel=2)@100, RR], qrels)

    name = os.path.basename(out_path).strip('.tsv')
    parent = os.path.dirname(out_path)

    bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25") % 1000

    rez = bm25.transform(new_queries).rename(columns={'qid' : 'query_id', 'docno' : 'doc_id'})
    rez.to_csv(os.path.join(parent, f'{name}_ranking.tsv'), sep='\t', index=False)

    metrics = evaluator.calc_aggregate(rez)
    metrics['name'] = name
    
    with open(os.path.join(parent, 'metrics.jsonl'), 'a') as f:
        f.write(str(metrics) + '\n')

if __name__ == "__main__":
    Fire(main)