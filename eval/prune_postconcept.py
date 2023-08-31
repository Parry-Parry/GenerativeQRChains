import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
from fire import Fire 
from lightchain import LambdaChain, Prompt
from conceptqr.chains.conceptextraction import NeuralExtraction
from conceptqr.chains.weighting import FixedWeighting
from conceptqr.chains.pruning import IDFPrune
from conceptqr.chains.conceptexpansion import ConceptExpansion
from conceptqr.models import LM
from conceptqr.models.generation import creative
from conceptqr.util import concatenate
from transformers import T5ForConditionalGeneration, T5Tokenizer
import ir_measures
from ir_measures import *

import torch

DATASET = 'irds:msmarco-passage/trec-dl-2019/judged'
INDEX_DATASET = 'msmarco_passage'
LM_NAME_OR_PATH = 'google/flan-t5-xxl'    
CONCEPTS = [1, 3, 5, 10]
TOPK = [5, 10]

def concatenate_concepts(inp):
    # group by qid and concatenate expansion_terms over concept columns
    inp = inp.groupby(['qid', 'query', 'concept'])['expansion_terms'].agg(list).reset_index()
    inp = inp.groupby(['qid', 'query'])['expansion_terms'].apply(lambda x: [term for terms in x for term in terms]).reset_index()
    inp['expansion_terms'] = inp['expansion_terms'].apply(lambda x : ' '.join(concatenate(x)))
    return inp

def sample_terms(x, k):
    import random
    x['expansion_terms'] = x['expansion_terms'].apply(lambda x : ' '.join(random.sample(x.split(' '), k)) if len(x.split(' ')) > k else x)
    return x

def main(out_file : str, batch_size : int = 16):
    ### INIT PIPE ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flan_kwargs = {'device_map' : 'auto', 'torch_dtype' : torch.float16}
    flan = T5ForConditionalGeneration.from_pretrained(LM_NAME_OR_PATH, **flan_kwargs)
    tokenizer = T5Tokenizer.from_pretrained(LM_NAME_OR_PATH)

    tok_kwargs = {
        'padding' : 'max_length', 'truncation' : True,
    }

    lm = LM(flan, tokenizer, generation_kwargs=creative, tokenizer_kwargs=tok_kwargs, batch_size=batch_size)
    extract = NeuralExtraction(lm, max_concepts=1)
    qr = ConceptExpansion(lm, "expansion_terms")

    ConceptConcatenation = LambdaChain(concatenate_concepts)

    downstream = qr >> ConceptConcatenation 

    ### INIT DATA ###

    dataset = pt.get_dataset(DATASET)
    topics = dataset.get_topics()
    qrels = dataset.get_qrels().rename(columns={'qid' : 'query_id', 'docno' : 'doc_id', 'label' : 'relevance'})
    evaluator = ir_measures.evaluator([nDCG@10, R(rel=2)@100, R(rel=2)@1000, P(rel=2)@10, P(rel=2)@100, RR, AP(rel=2)], qrels)

    bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25") % 1000

    queries = topics[['qid', 'query']].copy().drop_duplicates()

    ALL_RES = []

    ### RUN PIPE ###

    for k in TOPK:
        for c in CONCEPTS:
            weighting = FixedWeighting(100, 0.5)
            #sampler = LambdaChain(lambda x : sample_terms(x, 20))
            pruner = IDFPrune(INDEX_DATASET, topk=k)
            extract.max_concepts = c
            pipe = extract >> downstream >> pruner >> weighting
            concepts_expand = pipe(queries)
            ranking = bm25.transform(concepts_expand).rename(columns={'qid' : 'query_id', 'docno' : 'doc_id'})
            metrics = evaluator.calc_aggregate(ranking)

            metrics = {str(k) : v for k, v in metrics.items()}

            metrics['topk'] = k
            metrics['sample_topk'] = 20
            metrics['num_concepts'] = c

            ALL_RES.append(metrics)
    
    baseline = bm25.transform(topics).rename(columns={'qid' : 'query_id', 'docno' : 'doc_id'})
    metrics = evaluator.calc_aggregate(baseline)

    metrics = {str(k) : v for k, v in metrics.items()}
    metrics['topk'] = 0
    metrics['sample_topk'] = 0
    metrics['num_concepts'] = 0

    ALL_RES.append(metrics)

    ALL_RES = pd.DataFrame.from_records(ALL_RES)
    ALL_RES.to_csv(out_file, sep='\t', index=False)


if __name__ == "__main__":
    Fire(main)