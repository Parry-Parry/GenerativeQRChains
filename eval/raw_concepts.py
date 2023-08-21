import pyterrier as pt
if not pt.started():
    pt.init()

from fire import Fire
from conceptqr import LM, NeuralExtraction, ConceptExpansion
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def main(lm_name_or_path : str, 
         test_set : str,
         out_path : str,
         device_map=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flan_kwargs = {}
    if device_map is not None:
        flan_kwargs['device_map'] = device_map
    flan = T5ForConditionalGeneration.from_pretrained(lm_name_or_path, **flan_kwargs)
    if device_map is None:
        flan = flan.to(device)
    tokenizer = T5Tokenizer.from_pretrained(lm_name_or_path)

    lm = LM(flan, tokenizer, generation_kwargs={'max_length': 512})
    extract = NeuralExtraction(lm)
    qr = ConceptExpansion(lm, "expansion_terms")

    pipe = extract >> qr

    test = pt.get_dataset(test_set)
    topics = test.get_topics()

    queries = topics[['qid', 'query']].copy().drop_duplicates()
    concepts_expand = pipe(queries)

    concepts_expand.to_csv(out_path, sep='\t', index=False)

if __name__ == "__main__":
    Fire(main)