import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
from fire import Fire 
from lightchain import LambdaChain, Prompt
from conceptqr.chains.conceptextraction import NeuralExtraction
from conceptqr.chains.weighting import FixedWeighting
from conceptqr.chains.conceptexpansion import ConceptExpansion
from conceptqr.models import LM
from conceptqr.models.generation import creative
from conceptqr.util import concatenate
from transformers import T5ForConditionalGeneration, T5Tokenizer
import ir_measures
from ir_measures import *

import torch

DATASET = 'irds:msmarco-passage/trec-dl-2019/judged'
LM_NAME_OR_PATH = 'google/flan-t5-xl'    
CONCEPTS = [1, 3, 5]

PROMPTS = [
    "Query: {query} \n\n Extract list of main concepts from the query:",
    "Query: {query} \n\n Extract list of main topics from the query:",
    "Query: {query} \n\n What topics relate to this query:",
    "Query: {query} \n\n What concepts relate to this query:",
    "Query: {query} \n\n What topics are related to this query:",
    "Query: {query} \n\n What concepts are related to this query:",
]

def concatenate_concepts(inp):
    # group by qid and concatenate expansion_terms over concept columns
    inp = inp.groupby(['qid', 'query', 'concept'])['expansion_terms'].agg(list).reset_index()
    inp = inp.groupby(['qid', 'query'])['expansion_terms'].apply(lambda x: [term for terms in x for term in terms]).reset_index()
    inp['expansion_terms'] = inp['expansion_terms'].apply(lambda x : ' '.join(concatenate(x)))
    return inp

def main(out_file : str):
    ### INIT PIPE ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flan_kwargs = {}
    flan = T5ForConditionalGeneration.from_pretrained(LM_NAME_OR_PATH, **flan_kwargs)
    flan = flan.to(device)
    tokenizer = T5Tokenizer.from_pretrained(LM_NAME_OR_PATH)

    tok_kwargs = {
        'padding' : 'max_length', 'truncation' : True,
    }

    lm = LM(flan, tokenizer, generation_kwargs=creative, tokenizer_kwargs=tok_kwargs, batch_size=32)
    extract = NeuralExtraction(lm, max_concepts=1)
    qr = ConceptExpansion(lm, "expansion_terms")
    weighting = FixedWeighting(10, 0.5)
    ConceptConcatenation = LambdaChain(concatenate_concepts)

    downstream = qr >> ConceptConcatenation >> weighting

    ### INIT DATA ###

    dataset = pt.get_dataset(DATASET)
    topics = dataset.get_topics()
    qrels = dataset.get_qrels().rename(colums={'qid' : 'query_id', 'docno' : 'doc_id', 'label' : 'relevance'})
    evaluator = ir_measures.evaluator([nDCG@10, R(rel=2)@100, R(rel=2)@1000, P(rel=2)@10, P(rel=2)@100, RR], qrels)

    bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25") % 1000

    queries = topics[['qid', 'query']].copy().drop_duplicates()

    ALL_RES = []

    ### RUN PIPE ###

    for c in CONCEPTS:
        for i, prompt in enumerate(PROMPTS):
            extract.prompt = Prompt.from_string(prompt)
            extract.max_concepts = c
            pipe = extract >> downstream
            concepts_expand = pipe(queries)
            ranking = bm25.transform(concepts_expand).rename(columns={'qid' : 'query_id', 'docno' : 'doc_id'})
            metrics = evaluator.calc_aggregate(ranking)

            metrics = {str(k) : v for k, v in metrics.items()}

            metrics['num_concepts'] = c
            metrics['prompt_variant'] = i
            metrics['prompt_text'] = prompt

            ALL_RES.append(metrics)

    ALL_RES = pd.DataFrame.from_records(ALL_RES)
    ALL_RES.to_csv(out_file, sep='\t', index=False)


if __name__ == "__main__":
    Fire(main)