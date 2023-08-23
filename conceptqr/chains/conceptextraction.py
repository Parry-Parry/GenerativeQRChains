from lightchain import Chain, Prompt

ExtractionPrompt = Prompt.from_string("Query: {query} \n\n Extract comma seperated list of main concepts from the query:")
class NeuralExtraction(Chain):
    essential = ['query']
    def __init__(self, model, max_concepts : int = 3, out_attr : str = 'expansion_terms', i=0):
        super(NeuralExtraction, self).__init__(model=model, prompt=ExtractionPrompt, name=f"Neural_Extraction{i}")
        self.max_concepts = max_concepts
        self.out_attr = out_attr

    def post_process(self, inp):
        print(inp)
        split = inp.split(',')[:self.max_concepts]
        print(split)
        return [s.strip() for s in split]

    def logic(self, inp):
        assert 'query' in inp.columns, "Neural Extraction requires 'query' in input"
        out = inp.copy()
        prompt_args = inp[self.essential].to_dict(orient='records')
        prompts = self.prompt(prompt_args)
        out[self.out_attr] = list(map(lambda x : self.post_process(x[0]), self.model(prompts)))