
creative = {
    'min_length' : 20, 
    'max_new_tokens' : 40, 
    'temperature' : 1.2,
    'do_sample' : True,
    'top_k' : 200,
    'top_p' : 0.92,
    'repetition_penalty' : 2.1,
    'early_stopping' : True
}

deterministic = {
    'min_length' : 20, 
    'max_new_tokens' : 40, 
    'do_sample' : False,
    'top_k' : 200,
    'top_p' : 0.92,
    'repetition_penalty' : 2.1,
    'early_stopping' : True
}

contrastive = {
    'min_length' : 20,
    'max_new_tokens' : 40,
    'temperature' : 1.2,
    'do_sample' : True,
    'top_k' : 200,
    'top_p' : 0.92,
    'repetition_penalty' : 2.1,
    'early_stopping' : True,
    'penalty_alpha' : 0.5,
}

beam_search ={
    'min_length' : 20,
    'max_new_tokens' : 40,
    'temperature' : 1.2,
    'do_sample' : False,
    'top_k' : 50,
    'top_p' : 0.92,
    'repetition_penalty' : 2.1,
    'early_stopping' : True,
    'num_beams' : 5,
    'num_return_sequences' : 1,
    'no_repeat_ngram_size' : 1,
}