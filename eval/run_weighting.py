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
    topics = pd.read_csv(intermediate, sep='\t', index_col=False, converters={'expansion_terms': ast.literal_eval})

    new_queries = pipe(topics)
    new_queries.to_csv(out_path, sep='\t', index=False)

if __name__ == "__main__":
    Fire(main)