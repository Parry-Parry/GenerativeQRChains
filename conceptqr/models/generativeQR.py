import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
from lightchain import LambdaChain 

from conceptqr.chains.conceptexpansion import ConceptExpansion
from pandas import DataFrame

def concatenate_concepts(inp):
    # group by qid and concatenate expansion_terms over concept columns
    inp = inp.groupby(['qid', 'query', 'concept'])['expansion_terms'].agg(list).reset_index()
    return inp.groupby(['qid', 'query'])['expansion_terms'].apply(list).reset_index()

ConceptConcatenation = LambdaChain(concatenate_concepts, name="Concept_Concatenation")

class GenerativeConceptQR(pt.Transformer):
    def __init__(self, model, concept_extract, weighting_model) -> None:
        super().__init__()
        self.pipeline = concept_extract >> ConceptExpansion(model, "expansion_terms") >> ConceptConcatenation >> weighting_model
    
    def transform(self, inputs: DataFrame) -> DataFrame:
        queries = inputs[['qid', 'query']].copy().drop_duplicates()
        queries['new'] = self.pipeline(queries)
        return queries[['qid', 'query' 'new']].rename(columns = {'query' : 'query_0', 'new' : 'query'})
