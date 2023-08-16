from lightchain import Chain, Prompt

ExtractionPrompt = Prompt.from_string("Query: {query} \n\n Extract main concepts from the query:")
class ConceptExtraction(Chain):
    essential = ['query']
    def __init__(self, model, out_attr : str, i=0):
        super(ConceptExtraction, self).__init__(model=model, prompt=ExtractionPrompt, name=f"Concept_Extraction_{i}")
        self.out_attr = out_attr

    def logic(self, inp):
        assert 'query' in inp.columns, "ConceptExtraction requires 'query' in input"
        out = inp.copy()
        prompt_args = inp[self.essential].to_dict(orient='records')
        prompts = self.prompt(prompt_args)
        out[self.out_attr] = self.model(prompts)