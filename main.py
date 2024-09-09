import torch
import torchvision.transforms as transforms
import transformers
from transformers import (
    CLIPModel,
    CLIPTokenizer,
    CLIPProcessor,
    CLIPConfig,
    set_seed
)
from clip import clip

from data import create_dataset,create_loader
from computeID import computeID

from utils import *
from run_utils import *

from lora import run_lora
from loralib.layers import LinearLoRA

def main():

    # Load config file
    args = get_arguments()

    set_seed(args.seed)
    
    # CLIP
    if args.backbone == 'ViT-B/32':
        model_name_or_path  = '/d/lcx/airclip/model/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    config = CLIPConfig.from_pretrained(model_name_or_path)
    model.tokenize = clip.tokenize
    model.cuda()

    # Dataset
    train_dataset,val_dataset,test_dataset = create_dataset('retrieval_coco',args,None)
    samplers = [None,None,None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                                batch_size=[args.batch_size]*3,
                                                                num_workers=[args.num_workers]*3,
                                                                is_trains=[True, False, False], 
                                                                collate_fns=[None,None,None])  

    #Compute ID
    ids = computeID(args,model,train_loader,val_loader,test_loader,model.device)


    run_lora(args,model,train_loader,val_loader,test_loader,ids)
if __name__ == '__main__':
    main()