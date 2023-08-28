from fire import Fire
import os 
import subprocess as sp

def main(
        script : str,
        test_set : str,
        out_dir : str,
        weight_name_or_path : str = None,
        lm_name_or_path : str = None, 
        stopwords : str = None,
        batch_size : int = 8):
    
    BETA = [0.25, 0.5, 0.75, 1.0]
    TOPK = [1, 3, 5, 10, 20]
    MAX_CONCEPTS = [10]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    main_args = f'python {script} --lm_name_or_path {lm_name_or_path} --test_set {test_set} --stopwords {stopwords} --batch_size {batch_size}'
    if weight_name_or_path is not None:
        main_args += f' --weight_name_or_path {weight_name_or_path}'

    for beta in BETA:
        for topk in TOPK:
            for max_concepts in MAX_CONCEPTS:
                args = main_args
                out_path = os.path.join(out_dir, f'dl19_{beta}_{topk}_{max_concepts}.tsv')
                args += f' --beta {beta} --topk {topk} --max_concepts {max_concepts} --out_path {out_path}'
                sp.run(args, shell=True)
    
if __name__ == "__main__":
    Fire(main)
                

