import pyterrier as pt
if not pt.started():
    pt.init()

from fire import Fire
import os 
import pandas as pd
import ir_measures
import ir_datasets as irds
from ir_measures import *

def main(topic_dir : str, out_dir : str):
    files = [f for f in os.listdir(topic_dir) if os.path.isfile(os.path.join(topic_dir, f))]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25")
    
    runs = []

    qrels = irds.load("msmarco-passage/trec-dl-2019/judged").qrels_iter()
    evaluator = ir_measures.evaluator([nDCG@10, R(rel=2)@100, R(rel=2)@1000, P(rel=2)@10, P(rel=2)@100, RR], qrels)

    for f in files:
        name = os.path.basename(f).strip('.tsv')
        topics = pd.read_csv(os.path.join(topic_dir, f), sep='\t', index_col=False)
        rez = bm25.transform(topics)

        rez.to_csv(os.path.join(out_dir, f'{name}_ranking.tsv'), sep='\t', index=False)

        

        metrics = evaluator.calc_aggregate(rez)
        metrics['name'] = name
        runs.append(metrics)
    
    runs = pd.DataFrame.from_records(runs)
    runs.to_csv(os.path.join(out_dir, 'metrics.tsv'), sep='\t', index=False)

if __name__ == "__main__":
    Fire(main)