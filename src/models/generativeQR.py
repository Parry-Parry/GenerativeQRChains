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


        self.pipeline = concept_extract >> ConceptExpansion(model, "concept_terms") >> weighting_model
    
    def transform(self, inputs: DataFrame) -> DataFrame:
        queries = inputs[['qid', 'query']].copy().drop_duplicates()
        expansions = self.pipeline(queries)
        '''
            TODO:
            1. Choose how to merge the expansions with the original queries
        '''

        queries['new'] = queries['query'] + ' ' + expansions['new_terms']
        return queries[['qid', 'query' 'new']].rename(columns = {'query' : 'query_0', 'new' : 'query'})
