import pyterrier as pt
if not pt.started():
    pt.init()

from fire import Fire
from conceptqr.chains.conceptextraction import NeuralExtraction
from conceptqr.chains.conceptexpansion import ConceptExpansion
from conceptqr.models import LM
from conceptqr.models.generation import contrastive

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from conceptqr.util import concatenate
from lightchain import LambdaChain

def concatenate_concepts(inp):
    # group by qid and concatenate expansion_terms over concept columns
    inp = inp.groupby(['qid', 'query', 'concept'])['expansion_terms'].agg(list).reset_index()
    inp = inp.groupby(['qid', 'query'])['expansion_terms'].apply(lambda x: [term for terms in x for term in terms]).reset_index()
    inp['expansion_terms'] = inp['expansion_terms'].apply(lambda x : ' '.join(concatenate(x)))
    return inp

def main(lm_name_or_path : str, 
         test_set : str,
         out_path : str,
         device_map=None,
         max_concepts : int = 10,
         batch_size : int = 8):
    
    sub_cut = [1, 3, 5]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flan_kwargs = {}
    if device_map is not None:
        flan_kwargs['device_map'] = device_map
    flan = T5ForConditionalGeneration.from_pretrained(lm_name_or_path, **flan_kwargs)
    if device_map is None:
        flan = flan.to(device)
    tokenizer = T5Tokenizer.from_pretrained(lm_name_or_path)

    tok_kwargs = {
        'padding' : 'max_length', 'truncation' : True,
    }

    lm = LM(flan, tokenizer, generation_kwargs=contrastive, tokenizer_kwargs=tok_kwargs, batch_size=batch_size)
    extract = NeuralExtraction(lm, max_concepts=max_concepts)
    qr = ConceptExpansion(lm, "expansion_terms")

    ConceptConcatenation = LambdaChain(concatenate_concepts)

    pipe = extract >> qr 

    test = pt.get_dataset(test_set)
    topics = test.get_topics()

    queries = topics[['qid', 'query']].copy().drop_duplicates()
    concepts_expand = pipe(queries)
    concepts_expand.to_csv(out_path, sep='\t', index=False)

    for sub in sub_cut:
        concepts_expand_sub = concepts_expand.groupby('qid').head(sub)
        concepts_concat = ConceptConcatenation(concepts_expand_sub)
        concepts_concat.to_csv(out_path.replace('.tsv', f'_sub_{sub}.tsv'), sep='\t', index=False)
        
    concepts_concat = ConceptConcatenation(concepts_expand)
    concepts_concat.to_csv(out_path.replace('.tsv', '_concat.tsv'), sep='\t', index=False)

if __name__ == "__main__":
    Fire(main)