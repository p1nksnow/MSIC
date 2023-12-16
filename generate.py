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
from rouge import Rouge
from transformers import AutoTokenizer
import torch.distributed as dist
from tqdm import tqdm
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
warnings.filterwarnings("ignore")
import time

torch.manual_seed(1203)
np.random.seed(2048)
model_name = "HCModel"
data_path = "./data/records_final_mimic3.pkl"
voc_path = "./data/voc_final_mimic3.pkl"
log_dir  = "./log_our_model" 
saved_model = "xxx.pt"
checkpoint_dir = log_dir+"/"+saved_model

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
#parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--epoches", type=int, default=50, help="epoches")
parser.add_argument("--cuda", type=int, default=4, help="which cuda")

args = parser.parse_args()


def evaluate(model, dataset,voc_size):
    model.eval()
    preds_sym = []
    gts_sym = []
    preds_diag = []
    gts_diag = []
    preds_med = []
    gts_med = []
    ppl_logits = [[],[],[]]
    ppl_truths = [[],[],[]]
    visits = []
    during_time = 0
    with torch.no_grad():
        for idx, input in enumerate(tqdm(dataset)):
            start_time = time.time()
            outs = model(input,inference=True,only_pred=True)
            during_time += time.time() - start_time
            pp = []
            for i,item in enumerate(zip(outs[0][0],outs[1][0],outs[2][0],input)):
                pred_sym,pred_diag,pred_med,inp = item
                v = []
                if i!=0:
                    p=np.zeros(voc_size[0])
                    if len(pred_sym) > 0:
                        p[pred_sym]=1
                    preds_sym.append(p)
                    t=np.zeros(voc_size[0])
                    t[inp[0][0]]=1
                    gts_sym.append(t)


                    p=np.zeros(voc_size[1])
                    if len(pred_diag) > 0:
                        p[pred_diag]=1
                    preds_diag.append(p)
                    t=np.zeros(voc_size[1])
                    t[inp[0][1]]=1
                    gts_diag.append(t)
                    ppl_logits[1].append(outs[1][1][i-1,:])
                    ppl_logits[0].append(outs[0][1][i-1,:])
                    ppl_truths[1].append(inp[0][1])
                    ppl_truths[0].append(inp[0][0])

                if i == 0:
                    v.append(inp[0][0])
                    v.append(inp[0][1])
                    v.append(pred_med)
                else:
                    v.append(pred_sym)
                    v.append(pred_diag)
                    v.append(pred_med)

                pp.append(v)

                ppl_logits[2].append(outs[2][1][i,:]) 
                ppl_truths[2].append(inp[0][2])
                p=np.zeros(voc_size[2])
                if len(pred_med) > 0:
                    p[pred_med]=1
                preds_med.append(p)
                t=np.zeros(voc_size[2])
                t[inp[0][2]]=1
                gts_med.append(t)
            visits.append(pp)
    print(during_time)
    ppl_logits = [torch.stack(i,dim=0) for i in ppl_logits]
    dill.dump(visits,open("EHR/med_our_model/tmp/pred_visits_for_utility_mimic3",'wb'))
    

def generate(model, dataset, tokenizer):
    model.eval()
    bleu_list = []
    predict_text = []
    real_text = []
    with torch.no_grad():
        rouge = Rouge(metrics=["rouge-1", "rouge-2","rouge-l"])
        c_bleu, r_bleu, d_bleu=[0 for i in range(4)],[0 for i in range(4)],[0 for i in range(4)]
        c_rouge, r_rouge, d_rouge=[0 for i in range(3)],[0 for i in range(3)],[0 for i in range(3)]
        n_sentences =0
        c_no_s=0
        r_no_s=0
        d_no_s=0
        all_no_s=0
        all_bleu = [0 for i in range(4)]
        all_rouge= [0 for i in range(3)]
        f1=open(os.path.join(log_dir,f"his_output.txt"),'w')
        f2=open(os.path.join(log_dir,f"med_output.txt"),'w')
        for idx, input in enumerate(tqdm(dataset)):
            seqs,seq_lens = model(input,inference=True)
            seqs = [tokenizer.batch_decode(seq[:,seq_lens[i][0]:seq_lens[i][1]], skip_special_tokens=True)  for i,seq in enumerate(seqs)]
            n_sentences +=len(input)
            for inp, c_ans, r_ans, d_ans in zip(input,seqs[0],seqs[1],seqs[2]):
                c_gold=tokenizer.decode(inp[4][0], skip_special_tokens=True)
                r_gold=tokenizer.decode(inp[4][1], skip_special_tokens=True)
                d_gold=tokenizer.decode(inp[4][2], skip_special_tokens=True)
                
                f1.write('****Predict:****\n'+r_ans +'\n****GoLd:****\n' + r_gold+'\n\n\n')
                f2.write('****Predict:****\n'+d_ans +'\n****GoLd:****\n' + d_gold+'\n\n\n')
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
                
                all_gold = c_gold+" | "+r_gold+" | "+d_gold
                all_ans = c_ans + " | "+r_ans + " | "+d_ans

                all_bleu[0] += sentence_bleu([all_gold.split()], all_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[1,0,0,0])
                all_bleu[1] += sentence_bleu([all_gold.split()], all_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[0.5,0.5,0,0])
                all_bleu[2] += sentence_bleu([all_gold.split()], all_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[1/3,1/3,1/3,0])
                cc= sentence_bleu([all_gold.split()], all_ans.split(), smoothing_function=SmoothingFunction().method7, weights=[0.25,0.25,0.25,0.25])
                all_bleu[3] +=cc

                bleu_list.append(cc)
                predict_text.append(all_ans)
                real_text.append(all_gold)
                if len(all_ans) == 0 or len(all_gold) == 0  or all_gold == "." or all_ans == ".":
                    all_no_s +=1
                else:
                    all_r=rouge.get_scores(all_ans,all_gold)
                    all_rouge[0] += all_r[0]["rouge-1"]["f"]
                    all_rouge[1] += all_r[0]["rouge-2"]["f"]
                    all_rouge[2] += all_r[0]["rouge-l"]["f"]


        f1.close()
        f2.close()
        np.save(f"{log_dir}/bleu_list.npy",bleu_list)
        np.save(f"{log_dir}/predict_text.npy",predict_text)
        np.save(f"{log_dir}/real_text.npy",real_text)

        bleu_outs=[[i/n_sentences for i in c_bleu],[i/n_sentences for i in r_bleu],[i/n_sentences for i in d_bleu]]
        rouge_outs=[[i/(n_sentences-c_no_s) if n_sentences-c_no_s != 0 else 0 for i in c_rouge],[i/(n_sentences-r_no_s) if n_sentences-r_no_s != 0 else 0 for i in r_rouge],[i/(n_sentences-d_no_s) if n_sentences-d_no_s != 0 else 0 for i in d_rouge]]
        with open(os.path.join(log_dir,f"testresult_{saved_model}.txt"),'w') as t:
            t.write(f"chief_bleu:\n{[i/n_sentences for i in c_bleu]}\n")
            t.write(f"his_bleu:\n{[i/n_sentences for i in r_bleu]}\n")
            t.write(f"med_bleu:\n{[i/n_sentences for i in d_bleu]}\n")
            t.write(f"chief_rouge:\n{[i/(n_sentences-c_no_s) if n_sentences-c_no_s != 0 else 0 for i in c_rouge]}\n")
            t.write(f"his_rouge:\n{[i/(n_sentences-r_no_s) if n_sentences-r_no_s != 0 else 0 for i in r_rouge]}\n")
            t.write(f"med_rouge:\n{[i/(n_sentences-d_no_s) if n_sentences-d_no_s != 0 else 0 for i in d_rouge]}\n")
            t.write(f"all_rouge:\n{[i/(n_sentences-all_no_s) if n_sentences-all_no_s != 0 else 0 for i in all_rouge]}\n")
            t.write(f"all_bleu:\n{[i/n_sentences for i in all_bleu]}\n")

        
        
        return bleu_outs,rouge_outs

def main():
    # load data
    data = dill.load(open(data_path, "rb"))
    data = [i for i in data if len(i)<=5]
    voc = dill.load(open(voc_path, "rb"))
    sym_voc, diag_voc, med_voc = voc["sym_voc"], voc["diag_voc"],  voc["med_voc"]
    split_point = int(len(data) * 4 / 5)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]
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
    model.load_state_dict(torch.load(checkpoint_dir,map_location="cuda:{}".format(args.cuda)))
    generate(model, data_test, tokenizer)
         
  

if __name__ == "__main__":
    main()