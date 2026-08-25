import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = '1'



if __name__ == '__main__':

    device = '0'

    import random
    master_port = random.randint(1002,9999)
    
    nproc_per_node = len(device.split(','))
        
    run_yaml = f"CUDA_VISIBLE_DEVICES='{device}' torchrun --standalone --nproc_per_node 1 \
    --master_port {master_port} run.py --config_file YAML/bert4rec_swinbase.yaml"

    os.system(run_yaml)

 

