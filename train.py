import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import dill
import torch
from model import HealthConditionModel
import torch.nn.functional as F
from torch.optim import Adam
import argparse
import time
from transformers import AutoTokenizer
import transformers
import torch.distributed as dist
from tqdm import tqdm
import os
import wandb
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
import random
warnings.filterwarnings("ignore")

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)
model_name = "HCModel"
# load data
data_path = "./data/records_final_mimic3.pkl"
voc_path = "./data/voc_final_mimic3.pkl"
log_dir  = "./data/log_our_model"

log_output={
        "train/rc_loss":None,
        "train/kld_loss":None,
        "train/hcchief_loss":None,
        "train/hchis_loss":None,
        "train/hcmed_loss":None,
        "train/loss":None,
        "valid/hcchief_loss":None,
        "valid/hchis_loss":None,
        "valid/hcmed_loss":None,
        "valid/loss":None,
        "valid/chief_bleu1":None,
        "valid/chief_bleu2":None,
        "valid/chief_bleu3":None,
        "valid/chief_bleu4":None,
        "valid/his_bleu1":None,
        "valid/his_bleu2":None,
        "valid/his_bleu3":None,
        "valid/his_bleu4":None,
        "valid/med_bleu1":None,
        "valid/med_bleu2":None,
        "valid/med_bleu3":None,
        "valid/med_bleu4":None,
        "valid/chief_rouge1":None,
        "valid/chief_rouge2":None,
        "valid/chief_rougel":None,
        "valid/his_rouge1":None,
        "valid/his_rouge2":None,
        "valid/his_rougel":None,
        "valid/med_rouge1":None,
        "valid/med_rouge2":None,
        "valid/med_rougel":None,
        "learning_rate":None,
        }

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
#parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--end_lr", type=float, default=5e-7, help="learning rate")
parser.add_argument("--epoches", type=int, default=30, help="epoches")
parser.add_argument("--cuda", type=int, default=1, help="which cuda")
parser.add_argument("--wandb_project", type=str, default="clinc")
parser.add_argument("--warmup_steps", type=int, default=6000)
parser.add_argument("--loss1", type=float, default=0.0)
parser.add_argument("--loss2", type=float, default=0.0)
parser.add_argument("--loss3", type=float, default=0.0)
parser.add_argument("--loss4", type=float, default=0.0)
parser.add_argument("--loss5", type=float, default=0.0)
parser.add_argument("--after_train", type=bool, default=False)
parser.add_argument("--after_epoches", type=int, default=8)
parser.add_argument("--after_lr", type=float, default=5e-6)
parser.add_argument("--afterloss1", type=float, default=0.0)
parser.add_argument("--afterloss2", type=float, default=0.0)
parser.add_argument("--afterloss3", type=float, default=0.0)
parser.add_argument("--afterloss4", type=float, default=0.0)
parser.add_argument("--afterloss5", type=float, default=0.0)

args = parser.parse_args()
with open(os.path.join(log_dir,'valid_log.txt'),'w') as t:
    argsDict = args.__dict__
    for eachArg, value in argsDict.items():
        t.writelines(eachArg + ' : ' + str(value) + '\n')
    t.write("\n\n")



def evaluate(model, dataset, tokenizer, step):
    model.eval()
    with torch.no_grad():
        allloss=0
        c_loss =0
        r_loss =0
        d_loss =0
        rouge = Rouge(metrics=["rouge-1", "rouge-2","rouge-l"])
        c_bleu, r_bleu, d_bleu=[0 for i in range(4)],[0 for i in range(4)],[0 for i in range(4)]
        c_rouge, r_rouge, d_rouge=[0 for i in range(3)],[0 for i in range(3)],[0 for i in range(3)]
        n_sentences =0
        c_no_s=0
        r_no_s=0
        d_no_s=0
        for idx, input in enumerate(tqdm(dataset)):
            seqs,seq_lens = model(input,inference=True)
            seqs = [tokenizer.batch_decode(seq[:,seq_lens[i][0]:seq_lens[i][1]], skip_special_tokens=True)  for i,seq in enumerate(seqs)]
            n_sentences +=len(input)

            for inp, c_ans, r_ans, d_ans in zip(input,seqs[0],seqs[1],seqs[2]):
                c_gold=tokenizer.decode(inp[4][0], skip_special_tokens=True)
                r_gold=tokenizer.decode(inp[4][1], skip_special_tokens=True)
                d_gold=tokenizer.decode(inp[4][2], skip_special_tokens=True)

                c_bleu[0] += sentence_bleu([c_gold.split()], c_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[1,0,0,0])
                c_bleu[1] += sentence_bleu([c_gold.split()], c_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[0.5,0.5,0,0])
                c_bleu[2] += sentence_bleu([c_gold.split()], c_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[1/3,1/3,1/3,0])
                c_bleu[3] += sentence_bleu([c_gold.split()], c_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[0.25,0.25,0.25,0.25])

                r_bleu[0] += sentence_bleu([r_gold.split()], r_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[1,0,0,0])
                r_bleu[1] += sentence_bleu([r_gold.split()], r_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[0.5,0.5,0,0])
                r_bleu[2] += sentence_bleu([r_gold.split()], r_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[1/3,1/3,1/3,0])
                r_bleu[3] += sentence_bleu([r_gold.split()], r_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[0.25,0.25,0.25,0.25])
            
                d_bleu[0] += sentence_bleu([d_gold.split()], d_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[1,0,0,0])
                d_bleu[1] += sentence_bleu([d_gold.split()], d_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[0.5,0.5,0,0])
                d_bleu[2] += sentence_bleu([d_gold.split()], d_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[1/3,1/3,1/3,0])
                d_bleu[3] += sentence_bleu([d_gold.split()], d_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[0.25,0.25,0.25,0.25])

                if len(c_ans) == 0 or len(c_gold) == 0 or c_gold == len(c_gold)*"." or c_ans == len(c_ans)*".":
                    c_no_s +=1
                else:
                    c_r=rouge.get_scores(c_ans,c_gold)
                    c_rouge[0] += c_r[0]["rouge-1"]["f"]
                    c_rouge[1] += c_r[0]["rouge-2"]["f"]
                    c_rouge[2] += c_r[0]["rouge-l"]["f"]

                if len(r_ans) == 0 or len(r_gold) == 0  or r_gold == len(r_gold)*"." or r_ans == len(r_ans)*".":
                    r_no_s +=1
                else:
                    r_r=rouge.get_scores(r_ans,r_gold)
                    r_rouge[0] += r_r[0]["rouge-1"]["f"]
                    r_rouge[1] += r_r[0]["rouge-2"]["f"]
                    r_rouge[2] += r_r[0]["rouge-l"]["f"]

                if len(d_ans) == 0 or len(d_gold) == 0 or d_gold == len(d_gold)*"." or d_ans == len(d_ans)*".":
                    d_no_s +=1
                else:
                    d_r=rouge.get_scores(d_ans,d_gold)
                    d_rouge[0] += d_r[0]["rouge-1"]["f"]
                    d_rouge[1] += d_r[0]["rouge-2"]["f"]
                    d_rouge[2] += d_r[0]["rouge-l"]["f"]
            
            if idx==len(dataset)-1:  
                f_c=open(f"{log_dir}/output"+str(step)+'chief.txt', 'w')
                f_h=open(f"{log_dir}/output"+str(step)+'his.txt', 'w')
                f_d=open(f"{log_dir}/output"+str(step)+'med.txt', 'w')
                for inp, c_ans, r_ans, d_ans in zip(input,seqs[0],seqs[1],seqs[2]):
                    c_gold=tokenizer.decode(inp[3][0], skip_special_tokens=True)
                    r_gold=tokenizer.decode(inp[3][1], skip_special_tokens=True)
                    d_gold=tokenizer.decode(inp[3][2], skip_special_tokens=True)
                    f_c.write('****Predict:****\n'+c_ans +'\n****GoLd:****\n' + c_gold+'\n\n\n')
                    f_h.write('****Predict:****\n'+r_ans +'\n****GoLd:****\n' + r_gold+'\n\n\n')
                    f_d.write('****Predict:****\n'+d_ans +'\n****GoLd:****\n' + d_gold+'\n\n\n')
                f_c.close()
                f_h.close()
                f_d.close()

        bleu_outs=[[i/n_sentences for i in c_bleu],[i/n_sentences for i in r_bleu],[i/n_sentences for i in d_bleu]]
        rouge_outs=[[i/(n_sentences-c_no_s) if n_sentences-c_no_s != 0 else 0 for i in c_rouge],[i/(n_sentences-r_no_s) if n_sentences-r_no_s != 0 else 0 for i in r_rouge],[i/(n_sentences-d_no_s) if n_sentences-d_no_s != 0 else 0 for i in d_rouge]]
        loss_outs =[c_loss/(n_sentences-c_no_s) if n_sentences-c_no_s != 0 else 0,r_loss/(n_sentences-r_no_s) if n_sentences-r_no_s != 0 else 0,d_loss/(n_sentences-d_no_s) if n_sentences-d_no_s != 0 else 0,allloss/n_sentences]
        return bleu_outs,rouge_outs, loss_outs

def main():
    data = dill.load(open(data_path, "rb"))
    data = [i for i in data if len(i)<=5]
    voc = dill.load(open(voc_path, "rb"))
    sym_voc, diag_voc, med_voc = voc["sym_voc"], voc["diag_voc"],  voc["med_voc"]
    split_point = int(len(data) * 4 / 5)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    device = torch.device("cuda:{}".format(args.cuda))
    #device=torch.device("cpu:0")
    tokenizer = AutoTokenizer.from_pretrained("biobart-v2-base")
    vocab = tokenizer.vocab_size
    voc_size = (len(sym_voc.idx2word), len(diag_voc.idx2word),  len(med_voc.idx2word))
    model = HealthConditionModel(
        voc_size,
        emb_dim=256,
        device=device,
    ).to(device)
    #model.load_state_dict(torch.load('.pt'))
    
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    param_groups = [{
        'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad == True ],
        'weight_decay':
        0.01
    }, {
        'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)) and p.requires_grad == True],
        'weight_decay':
        0.0
    }]
    
    optimizer = Adam(param_groups, lr=args.lr, eps=1e-8)
    total_steps = args.epoches * len(data_train)
    scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,lr_end=args.end_lr
    )
    global_step = 0

    wandb_config = dict(
        learning_rate = args.lr,
        max_epoch = args.epoches,
        model = str(model),
        optimizer = str(optimizer),
    )
    wandb.init(
        project=args.wandb_project,
        config=wandb_config,
        name=log_dir[-8:],
        )

    for epoch in range(args.epoches):
        print("\nepoch {} --------------------------{}".format(epoch + 1,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        model.train()
        for step, input in enumerate(tqdm(data_train)):
            global_step+=1
            loss1,loss2,loss3,loss4,loss5 = model(input)
            loss = args.loss1*loss1+ args.loss2*loss2+ args.loss3*loss3+ args.loss4*loss4+ args.loss5*loss5
            train_loss_log = [args.loss1*loss1.item(),args.loss2*loss2.item(),args.loss3*loss3.item(),args.loss4*loss4.item(),args.loss5*loss5.item(),loss.item()]
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if (global_step+1)%(2*split_point) == 0 and (global_step+1)>=(16*split_point) :
                bleus,rouges,valid_loss = evaluate(model, data_eval, tokenizer, global_step)
                log_file = os.path.join(log_dir, 'model{}_{}_{}_{}_{}_{}_{}.pt'.format(global_step,round(bleus[0][3],4),round(bleus[1][3],4),round(bleus[2][3],4),round(rouges[0][2],4),round(bleus[1][2],4),round(bleus[2][2],4)))
                torch.save(model.state_dict(),log_file)
                model.train()
                log_valid_loss= [-1*l for l in valid_loss]
                log_bleus = [i for b in bleus for i in b]
                log_rouges = [i for r in rouges for i in r]
                log_output_val = train_loss_log + log_valid_loss+log_bleus+log_rouges+[scheduler.get_last_lr()[0]]
                with open(os.path.join(log_dir,'valid_log.txt'),'a') as t:
                    t.write(f'\n\n\n\nsteps:{global_step}\n{log_bleus}\n{log_rouges}\n\n\n\n')
            else:
                log_output_val = train_loss_log+ [None]*25 + [scheduler.get_last_lr()[0]] 
            wandb.log(dict(zip(log_output.keys(),log_output_val)))
    if args.after_train:
        after_step=0
        model.train()
        for epoch in range(args.after_epoches):
            for k,v in model.named_parameters():
                v.requires_grad = False
            for k,v in model.hcdecoder.his_generator.named_parameters():
                v.requires_grad = True
            for k,v in model.hcdecoder.encoder_inpt_lm_his.named_parameters():
                v.requires_grad = True
            
            no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
            param_groups = [{
                'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad == True ],
                'weight_decay':
                0.01
            }, {
                'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)) and p.requires_grad == True],
                'weight_decay':
                0.0
            }] 
            optimizer1 = Adam(param_groups, lr=args.after_lr, eps=1e-8)
            total_steps = args.after_epoches * len(data_train)
            print("\nepoch {} --------------------------{}".format(epoch + 1,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            model.train()
            for step, input in enumerate(tqdm(data_train)):
                after_step+=1
                loss1,loss2,loss3,loss4,loss5 = model(input)
                loss = args.afterloss1*loss1+ args.afterloss2*loss2+args.afterloss3*loss3+args.afterloss4*loss4+ args.afterloss5*loss5
                train_loss_log = [args.afterloss1*loss1.item(),args.afterloss2*loss2.item(),args.afterloss3*loss3.item(),args.afterloss4*loss4.item(),args.afterloss5*loss5.item(),loss.item()]
                loss.backward()
                optimizer1.step()
                model.zero_grad()
                if (after_step+1)%(2*split_point) == 0 and (global_step+1)>=(8*split_point):
                    bleus,rouges,valid_loss = evaluate(model, data_eval, tokenizer, args.epoches*split_point+after_step)
                    log_file = os.path.join(log_dir, 'aftermodel{}_{}_{}_{}_{}_{}_{}.pt'.format(after_step,round(bleus[0][3],4),round(bleus[1][3],4),round(bleus[2][3],4),round(rouges[0][2],4),round(bleus[1][2],4),round(bleus[2][2],4)))
                    torch.save(model.state_dict(),log_file)
                    model.train()
                    log_valid_loss= [-1*l for l in valid_loss]
                    log_bleus = [i for b in bleus for i in b]
                    log_rouges = [i for r in rouges for i in r]
                    log_output_val = train_loss_log + log_valid_loss+log_bleus+log_rouges+[optimizer1.state_dict()['param_groups'][0]['lr']]
                    with open(os.path.join(log_dir,'valid_log.txt'),'a') as t:
                        t.write(f'\n\n\n\nsteps:{after_step}\n{log_bleus}\n{log_rouges}\n\n\n\n')
                else:
                    log_output_val = train_loss_log+ [None]*25 + [optimizer1.state_dict()['param_groups'][0]['lr']] 
                wandb.log(dict(zip(log_output.keys(),log_output_val)))

    
    wandb.finish()

if __name__ == "__main__":
    main()