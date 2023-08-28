from fire import Fire
import os 
import subprocess as sp

def main(
        script : str,
        run_dir : str,
        out_dir : str,):
    
    BETA = [0.25, 0.5, 0.75, 1.0]
    TOPK = [1, 3, 5, 10]
    PRUNE_K = [0, 10, 15, 20]

    weights = [ 'tfidf', 'fixed']
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    files = [os.path.join(run_dir, file) for file in os.listdir(run_dir) if file.endswith('.tsv')]
    
    main_args = f'python {script}'
    
    for file in files:
        name = os.path.basename(file).replace('.tsv', '')
        for prune_k in PRUNE_K:
            for weight in weights:
                    for topk in TOPK:
                        if weight == 'fixed' or weight == 'cwprf':
                            for beta in BETA:
                                args = main_args
                                out_path = os.path.join(out_dir, f'{name}_{beta}_{topk}_{prune_k}_{weight}.tsv')
                                args += f' --mode {weight} --intermediate {file} --beta {beta} --topk {topk} --out_path {out_path}'
                                if prune_k > 0: args += f' --prune_k {prune_k}'
                                sp.run(args, shell=True)
                        else:
                            args = main_args
                            out_path = os.path.join(out_dir, f'{name}_{0.0}_{topk}_{prune_k}_{weight}.tsv')
                            args += f' --mode {weight} --intermediate {file} --topk {topk} --out_path {out_path}'
                            if prune_k > 0: args += f' --prune_k {prune_k}'
                            sp.run(args, shell=True)
    
if __name__ == "__main__":
    Fire(main)
                

