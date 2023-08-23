from lightchain import Chain, Prompt

#ConceptPrompt = Prompt.from_string("Improve the search effectiveness by suggesting expansion terms for the query focusing on the concept {concept}:\n query:{query} \n expansion terms:")
ConceptPrompt = Prompt.from_string("Query: {query} \n\n Improve the search effectiveness by suggesting expansion terms for the query focusing on the concept of {concept}:")
class ConceptExpansion(Chain):
    essential = ['query', 'concept']
    def __init__(self, model, out_attr : str, i=0):
        super(ConceptExpansion, self).__init__(model=model, prompt=ConceptPrompt, name=f"Concept_Expansion_{i}")
        self.out_attr = out_attr

    def logic(self, inp):
        assert 'query' in inp and 'concept' in inp.columns, "ConceptExpansion requires 'query' and 'concept' in input"
        out = inp.copy()
        prompt_args = inp[self.essential].to_dict(orient='records')
        prompts = self.prompt(prompt_args)
        out[self.out_attr] = [*map(lambda x : x.split(' '), self.model(prompts))]
        return out