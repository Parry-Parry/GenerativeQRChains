import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
from lightchain import LambdaChain 

from conceptqr.chains.conceptexpansion import ConceptExpansion
from conceptqr.util import concatenate
from pandas import DataFrame

def concatenate_concepts(inp):
    # group by qid and concatenate expansion_terms over concept columns
    inp = inp.groupby(['qid', 'query', 'concept'])['expansion_terms'].agg(list).reset_index()
    inp = inp.groupby(['qid', 'query'])['expansion_terms'].apply(lambda x: [term for terms in x for term in terms]).reset_index()
    inp['expansion_terms'] = inp['expansion_terms'].apply(lambda x : ' '.join(concatenate(x)))
    return inp

ConceptConcatenation = LambdaChain(concatenate_concepts)

class GenerativeConceptQR(pt.Transformer):
    def __init__(self, model, concept_extract, weighting_model) -> None:
        self.pipeline = concept_extract >> ConceptExpansion(model, "expansion_terms") >> ConceptConcatenation >> weighting_model
    
    def transform(self, inputs: DataFrame) -> DataFrame:
        queries = inputs[['qid', 'query']].copy().drop_duplicates()
        queries = self.pipeline(queries)
        return queries[['qid', 'query' 'query_0']]
