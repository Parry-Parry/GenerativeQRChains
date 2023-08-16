import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()

from ..chains.conceptexpansion import ConceptExpansion
from pandas import DataFrame

class GenerativeConceptQR(pt.Transformer):
    def __init__(self, model, concept_extract, weighting_model) -> None:
        super().__init__()
        self.model = model
        self.concept_extract = concept_extract
        self.weighting_model = weighting_model

        self.pipeline = concept_extract >> ConceptExpansion(model, "expansion_terms") >> weighting_model
    
    def transform(self, inputs: DataFrame) -> DataFrame:
        queries = inputs[['qid', 'query']].copy().drop_duplicates()
        queries['new'] = self.pipeline(queries)
        return queries[['qid', 'query' 'new']].rename(columns = {'query' : 'query_0', 'new' : 'query'})
