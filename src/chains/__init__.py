from lightchain import LambdaChain 

def concatenate_concepts(inp):
    # group by qid and concatenate expansion_terms over concept columns
    return inp.groupby(['qid', 'query'])['expansion_terms'].apply(lambda x: ','.join(x)).reset_index()

ConceptConcatenation = LambdaChain(concatenate_concepts, name="Concept_Concatenation")