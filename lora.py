import torch
import torch.nn.functional as F
from tqdm import tqdm

from computeID import computeID
from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
from loralib import LinearLoRA

@torch.no_grad()
def evaluation(model, data_loader, device):
    # test
    model.eval() 
    
    print('Computing features for evaluation...')

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []  
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            for i in range(0, num_text, text_bs):
                text = texts[i: min(num_text, i+text_bs)]
                text_input = model.tokenize(text).to(device) 
                text_output = model.get_text_features(text_input)
                text_embed = text_output / text_output.norm(dim=1, keepdim=True)
                text_embeds.append(text_embed)   
            text_embeds = torch.cat(text_embeds,dim=0)

            image_embeds = []
            for image, img_id in data_loader: 
                image = image.to(device) 
                image_feat = model.get_image_features(image)
                image_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
                image_embeds.append(image_embed)
            image_embeds = torch.cat(image_embeds,dim=0)

            sims_matrix = image_embeds @ text_embeds.t()
    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


def run_lora(args,model,train_loader,val_loader,test_loader,ids):
    list_lora_layers = apply_lora(args,model)
    model.cuda()
    
    if args.eval_only:
        #waitting to fix
        # load_lora(args, list_lora_layers)
        # score_test_i2t, score_test_t2i = evaluation(model, test_loader, model.device)
        # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
        # print(test_result)
        return

    # make only parms to trian
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(get_lora_parameters(model), weight_decay=args.weight_decay, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-6)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    best_r1_val,best_r1_test=0.,0.
    best_epoch_val = 0

    # Guide the LoRA when to merge the parameters
    idcator = ids[142]
    for epoch in range(1,args.epoch + 1):
            model.train()
            loss_epoch = 0.

    
            for i,(image,caption,idx) in enumerate(tqdm(train_loader)):
                image = image.cuda()

                
                text = model.tokenize(caption).cuda()

                output = model(pixel_values=image,input_ids = text)
                ground_truth = torch.arange(len(output.logits_per_image)).long().cuda()
                loss = (
                            loss_img(output.logits_per_image, ground_truth)
                            + loss_txt(output.logits_per_text, ground_truth)
                        ) / 2

                    
                loss_epoch += loss.item() * image.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            print('Epoch :{}  Loss: {:.4f} lr: {:.6f}'.format(epoch,loss_epoch/len(train_loader.dataset),scheduler.get_last_lr()[0]))

            model.eval()
            score_val_i2t, score_val_t2i, = evaluation(model, val_loader, model.device)
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            txt_r1 = val_result['txt_r1']
            img_r1 = val_result['img_r1']

            print(f'tr@1:{txt_r1:.2f}, rr@1:{img_r1:.2f}')
            score_test_i2t, score_test_t2i = evaluation(model, test_loader, model.device)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
            test_result['epoch'] = epoch
            print(test_result)

            if args.cola:
                for n,m in model.named_modules():
                    if type(m) == LinearLoRA:
                        m.merge_BA()
                        m.init_lora_param()
                
                model.cuda()
                # reset optimizer
                optimizer = torch.optim.AdamW(get_lora_parameters(model), weight_decay=args.weight_decay, lr=args.lr)

            if args.idcola:
                new_ids = computeID(args,model,train_loader,val_loader,test_loader,model.device)
                new_idcator = new_ids[142]
                if new_idcator < idcator:
                    idcator = new_idcator
                    for n,m in model.named_modules():
                        if type(m) == LinearLoRA:
                            m.merge_lora_param()
                            m.init_lora_param()

                    model.cuda()
                    # reset optimizer
                    optimizer = torch.optim.AdamW(get_lora_parameters(model), weight_decay=args.weight_decay, lr=args.lr)
                    
    if args.save_path != None:
        #waitting to fixxx
        #save_lora(args,list_lora_layers)
        return
