
import random
import argparse  
import numpy as np 
import torch



def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default='retrieval_coco')
    parser.add_argument("--ann_root", type=str, default="annotations")
    parser.add_argument("--image_root", type=str, default="/s/COCO2014")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/32', type=str)
    # Training arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v','proj','fc1','fc2'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=32, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--lora_alpha', default=1, type=float, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    

    # ID arguments
    parser.add_argument('-n', '--nsamples', type=int, default=2000,help='Number of samples selected')
    parser.add_argument('--cpu', action='store_true', help="CPU mode")
    parser.add_argument('-s', action='store_true', help="skip data extract")  # on/off flag
    parser.add_argument('--Path', default='test/base', help="Paths for storing intermediate and final results")
    parser.add_argument('--update',action="store_true",help="choose whether to update the model")
    parser.add_argument('--use_delta',action="store_true",help="use deltaW to update")
    
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    args = parser.parse_args()

    return args
    

        
