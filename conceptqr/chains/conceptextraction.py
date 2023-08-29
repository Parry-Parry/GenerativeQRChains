from lightchain import Chain, Prompt

ExtractionPrompt = Prompt.from_string("query: {query} \n\n What concepts are related to the query:")
class NeuralExtraction(Chain):
    essential = ['query']
    def __init__(self, model, max_concepts : int = 3, out_attr : str = 'concept', split_token=' ', i=0):
        super(NeuralExtraction, self).__init__(model=model, prompt=ExtractionPrompt, name=f"Neural_Extraction{i}")
        self.max_concepts = max_concepts
        self.out_attr = out_attr
        self.split_token = split_token

    def post_process(self, inp):
        split = inp.split(self.split_token)[:self.max_concepts]
        return [s.strip() for s in split]

    def logic(self, inp):
        assert 'query' in inp.columns, "Neural Extraction requires 'query' in input"
        out = inp.copy()
        prompt_args = inp[self.essential].to_dict(orient='records')
        prompts = self.prompt(prompt_args)
        output = self.model(prompts)
        out[self.out_attr] = list(map(lambda x : self.post_process(x), output))
        out = out.explode(self.out_attr)
        return out