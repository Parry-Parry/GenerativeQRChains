from lightchain import Object 
from abc import abstractmethod

class ConceptExtractor(Object):
    essential = ['qid', 'query']
    def __init__(self, model):
        super(ConceptExtractor, self).__init__(name="Concept_Extractor")
        self.model = model

    @abstractmethod
    def parse_concepts(self, inp):
        raise NotImplementedError
    
    def __call__(self, inp):
        out = inp.drop_duplicates(self.essential).copy()
        out['concept'] = self.parse_concepts(out)
        out = out.explode('concept')
        return out
        
        


