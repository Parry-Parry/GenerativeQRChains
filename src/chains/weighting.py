from lightchain import Object 
from abc import abstractmethod

class WeightingModel(Object):
    essential = ['qid', 'expansion_terms']
    def __init__(self):
        super(WeightingModel, self).__init__(name="Weighting_Model")

    @abstractmethod
    def weight_terms(self, inp):
        raise NotImplementedError
    
    def __call__(self, inp):
        out = inp.copy()
        out['weighted_terms'] = self.weight_terms(out)
        return out
        
        


