import argparse 

from evaluation.models.base_model import convert_fsdp_checkpoints_to_hfmodels
from transformers import AutoTokenizer, AutoConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('local_dir', type=str, help='Path to the local directory containing the FSDP checkpoints.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory where the converted HF models will be saved.')
    parser.add_argument('--huggingface_dir', type=str, default='/mnt/petrelfs/jiangshuyang/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B')
    
    
    args = parser.parse_args()
    
    convert_fsdp_checkpoints_to_hfmodels(args.local_dir, args.output_dir, args.huggingface_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_dir)
    tokenizer.save_pretrained(args.output_dir)
    