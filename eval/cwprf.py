import pyterrier as pt
if not pt.started():
    pt.init()

from fire import Fire
from conceptqr import LM, GenerativeConceptQR, CWPRF_Weighting, NeuralExtraction
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def main(weight_name_or_path : str, 
         lm_name_or_path : str, 
         test_set : str,
         out_path : str,
         stopwords : str = None,
         beta : float = 0.5,
         topk : int = 20,):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flan_kwargs = {}
    flan = T5ForConditionalGeneration.from_pretrained(lm_name_or_path, **flan_kwargs)
    tokenizer = T5Tokenizer.from_pretrained(lm_name_or_path)

    lm = LM(flan, tokenizer, generation_kwargs={'max_length': 512})
    extract = NeuralExtraction(lm)
    cwprf = CWPRF_Weighting(weight_name_or_path, 
                            topk=topk, 
                            beta=beta, 
                            stopwords=True if stopwords else False, 
                            stopword_path = stopwords, 
                            weight_mode=None, 
                            id_mode='mean', 
                            device=device)
    qr = GenerativeConceptQR(lm, extract, cwprf)

    test = pt.get_dataset(test_set)
    topics = test.get_topics()

    new_queries = qr.transform(topics)
    new_queries.to_csv(out_path, sep='\t', index=False)

if __name__ == "__main__":
    Fire(main)