import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from transformers import AutoConfig,BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers import GenerationConfig

class BartGenerator(nn.Module):
    def __init__(self,generator_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained("./biobart-v2-base/config.json")
        self.model = BartForConditionalGeneration.from_pretrained(config=self.config,pretrained_model_name_or_path ="./biobart-v2-base/pytorch_model.bin")
        self.generator_name =generator_name #chief his med
        self.fix_parameters()

    def fix_parameters(self):
        for name,param in self.model.named_parameters():
            param.requires_grad =False # mask all generator parameters

    def forward(self,encoder_outputs,decoder_input_ids,labels):
        return self.model.forward(
            encoder_outputs=BaseModelOutput(encoder_outputs),
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
            ).loss

    def generate(self,encoder_outputs,last_outs=None): #chi_outs his_outs : (bsz,seq_len)
        #do sample
        config3=GenerationConfig(
            max_length=490,
            do_sample=False,
            num_beams=1,
            min_length=20,
            length_penalty=0,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            )
        config2=GenerationConfig(
            max_length=490,
            do_sample=True,
            num_beams=1,
            min_length=20,
            length_penalty=0,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            )
        config1=GenerationConfig(
            max_length=20,
            do_sample=False,
            num_beams=1,
            length_penalty=0,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            )
        

        if self.generator_name == "chief":
            if last_outs is None:
                outs = self.model.generate(
                    inputs_ids=None,
                    generation_config=config1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    encoder_outputs=BaseModelOutput(encoder_outputs)
                )
            else:
                d_enc_ins = last_outs[0][:,:490].masked_fill(last_outs[0][:,:490]==self.config.eos_token_id,self.config.pad_token_id)
                m_enc_ins = last_outs[1][:,:490].masked_fill(last_outs[1][:,:490]==self.config.eos_token_id,self.config.pad_token_id)
                bos = torch.ones((m_enc_ins.shape[0],1),dtype=torch.long,device=m_enc_ins.device)*self.config.bos_token_id
                m_enc_ins = torch.cat([bos,d_enc_ins[:,1:],m_enc_ins[:,1:]],dim=1)
                m_enc_outs = self.model.model.encoder(
                    input_ids = m_enc_ins
                )
                encoder_outputs = torch.cat([encoder_outputs,m_enc_outs[0][:,0,:].unsqueeze(1)],dim=1)
                outs = self.model.generate(
                    inputs_ids=None,
                    generation_config=config1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    encoder_outputs=BaseModelOutput(encoder_outputs)
                )
            
        elif self.generator_name == "his":
            d_enc_ins = last_outs[0][:,:20].masked_fill(last_outs[0][:,:20]==self.config.eos_token_id,self.config.pad_token_id)
            bos = torch.ones((d_enc_ins.shape[0],1),dtype=torch.long,device=d_enc_ins.device)*self.config.bos_token_id
            d_enc_ins = torch.cat([bos,d_enc_ins[:,1:]],dim=1)
            d_enc_outs = self.model.model.encoder(
                input_ids = d_enc_ins
            )
            encoder_outputs = torch.cat([encoder_outputs,d_enc_outs[0][:,0,:].unsqueeze(1)],dim=1)
            outs = self.model.generate(
                inputs_ids=None,
                generation_config=config2,
                output_scores=True,
                return_dict_in_generate=True,
                encoder_outputs=BaseModelOutput(encoder_outputs)
            )
        else:
            d_enc_ins = last_outs[0][:,:20].masked_fill(last_outs[0][:,:20]==self.config.eos_token_id,self.config.pad_token_id)
            m_enc_ins = last_outs[1][:,:490].masked_fill(last_outs[1][:,:490]==self.config.eos_token_id,self.config.pad_token_id)
            bos = torch.ones((m_enc_ins.shape[0],1),dtype=torch.long,device=m_enc_ins.device)*self.config.bos_token_id
            m_enc_ins = torch.cat([bos,d_enc_ins[:,1:],m_enc_ins[:,1:]],dim=1)
            m_enc_outs = self.model.model.encoder(
                input_ids = m_enc_ins
            )
            encoder_outputs = torch.cat([encoder_outputs,m_enc_outs[0][:,0,:].unsqueeze(1)],dim=1)
            outs = self.model.generate(
                input_ids=None,
                generation_config=config3,
                output_scores=True,
                return_dict_in_generate=True,
                encoder_outputs=BaseModelOutput(encoder_outputs)
            )

        return outs
    


class Multi_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_inpt_lm_chief = nn.Linear(256,768,bias=False)
        self.encoder_inpt_lm_his = nn.Linear(256,768,bias=False)
        self.encoder_inpt_lm_med = nn.Linear(256,768,bias=False)
        self.chi_generator = BartGenerator("chief")
        self.his_generator = BartGenerator("his")
        self.med_generator = BartGenerator("med")
        

    def forward(self, hc, sym, dis, med, label,inps,enc_ins):
        hc = hc.squeeze(0).unsqueeze(1)
        sym = sym.squeeze(0).unsqueeze(1)
        dis = dis.squeeze(0).unsqueeze(1)
        med = med.squeeze(0).unsqueeze(1)
        chief_label, his_label, med_label = label
        chief_inp, his_inp, med_inp = inps
        
        chief_label = chief_label.to(hc.device)
        his_label = his_label.to(hc.device)
        med_label = med_label.to(hc.device)
        chief_inp = chief_inp.to(hc.device)
        his_inp = his_inp.to(hc.device)
        med_inp = med_inp.to(hc.device)
        d_enc_in = enc_ins[0].to(hc.device)
        m_enc_in = enc_ins[1].to(hc.device)
        s_enc_in = enc_ins[2].to(hc.device)
        
        d_enc_outs = self.his_generator.model.model.encoder(
            input_ids = d_enc_in,
            return_dict = False
        )
        m_enc_outs = self.med_generator.model.model.encoder(
            input_ids = m_enc_in,
            return_dict = False
        )
        s_enc_outs = self.med_generator.model.model.encoder(
            input_ids = s_enc_in,
            return_dict = False
        )

        input1 = self.encoder_inpt_lm_chief(torch.cat([hc, sym],dim=1))
        input2 = self.encoder_inpt_lm_his(torch.cat([hc, sym, dis],dim=1))
        input3 = self.encoder_inpt_lm_med(torch.cat([hc, sym, dis, med],dim=1)) #(bsz,4,768)
        
        input2 = torch.cat([input2,d_enc_outs[0][:,0,:].unsqueeze(1)],dim=1)
        input3 = torch.cat([input3,m_enc_outs[0][:,0,:].unsqueeze(1)],dim=1)
        input1 = torch.cat([input1,s_enc_outs[0][:,0,:].unsqueeze(1)],dim=1)

        chief_loss = self.chi_generator.forward(encoder_outputs=input1,decoder_input_ids=chief_inp,labels=chief_label)
        his_loss = self.his_generator.forward(encoder_outputs=input2,decoder_input_ids=his_inp,labels=his_label)
        med_loss = self.med_generator.forward(encoder_outputs=input3,decoder_input_ids=med_inp,labels=med_label)

        return chief_loss,his_loss,med_loss

    def generate(self, h, s, d, m):
        h = h.squeeze(0).unsqueeze(1)
        s = s.squeeze(0).unsqueeze(1)
        d = d.squeeze(0).unsqueeze(1)
        m = m.squeeze(0).unsqueeze(1)

        input1 = self.encoder_inpt_lm_chief(torch.cat([h, s],dim=1))
        input2 = self.encoder_inpt_lm_his(torch.cat([h, s, d],dim=1))
        input3 = self.encoder_inpt_lm_med(torch.cat([h, s, d, m],dim=1))

        chi_outs = self.chi_generator.generate(input1)
        his_outs = self.his_generator.generate(input2,[chi_outs.sequences])
        chi_his_outs = [chi_outs.sequences,his_outs.sequences]
        med_outs = self.med_generator.generate(input3,chi_his_outs)
        his_med_outs = [his_outs.sequences,med_outs.sequences]
        chi_outs = self.chi_generator.generate(input1,his_med_outs)

        seqs = [chi_outs.sequences,his_outs.sequences,med_outs.sequences]
        seqs_lens = [(0,chi_outs.sequences.shape[-1]),(0,his_outs.sequences.shape[-1]),(0,med_outs.sequences.shape[-1])]
        return seqs,seqs_lens





class PriorNet(nn.Module):
    r""" 计算先验概率p(z|x)的网络, x为解码器最后一步的输出 """
    def __init__(self, x_size,  # post编码维度
                 latent_size,  # 潜变量维度
                 dims):  # 隐藏层维度
        super(PriorNet, self).__init__()
        assert len(dims) >= 1  # 至少两层感知机

        dims = [x_size] + dims + [latent_size*2]
        dims_input = dims[:-1]
        dims_output = dims[1:]

        self.latent_size = latent_size
        self.mlp = nn.Sequential()
        for idx, (x, y) in enumerate(zip(dims_input[:-1], dims_output[:-1])):
            self.mlp.add_module(f'linear{idx}', nn.Linear(x, y))  # 线性层
            self.mlp.add_module(f'activate{idx}', nn.Tanh())  # 激活层
        self.mlp.add_module('output', nn.Linear(dims_input[-1], dims_output[-1]))

    def forward(self, x):  # [batch, x_size]
        predict = self.mlp(x)  # [batch, latent_size*2]
        mu, logvar = predict.split([self.latent_size]*2, 1)
        return mu,  logvar

class RecognizeNet(nn.Module):
    r""" 计算后验概率p(z|x,y)的网络;x,y为解码器最后一步的输出 """
    def __init__(self, x_size,  # post编码维度
                 y_size,  # response编码维度
                 latent_size,  # 潜变量维度
                 dims):  # 隐藏层维度
        super(RecognizeNet, self).__init__()
        assert len(dims) >= 1  # 至少两层感知机

        dims = [x_size+y_size] + dims + [latent_size*2]
        dims_input = dims[:-1]
        dims_output = dims[1:]

        self.latent_size = latent_size
        self.mlp = nn.Sequential()
        for idx, (x, y) in enumerate(zip(dims_input[:-1], dims_output[:-1])):
            self.mlp.add_module(f'linear{idx}', nn.Linear(x, y))  # 线性层
            self.mlp.add_module(f'activate{idx}', nn.Tanh())  # 激活层
        self.mlp.add_module('output', nn.Linear(dims_input[-1], dims_output[-1]))

    def forward(self, x,  # [batch, x_size]
                y):  # [batch, y_size]
        x = torch.cat([x, y], 1)  # [batch, x_size+y_size]
        predict = self.mlp(x)  # [batch, latent_size*2]
        mu, logvar = predict.split([self.latent_size]*2, 1)
        return mu, logvar


class HealthConditionModel(nn.Module):
    def __init__(
        self,
        voc_size,
        emb_dim=256,
        device=torch.device("cpu:0"),
    ):
        super(HealthConditionModel, self).__init__()

        self.device = device
        self.embeddings = nn.ModuleList(
            [nn.Embedding(voc_size[i], emb_dim) for i in range(3)]
        )
        self.dropout = nn.Dropout(p=0.5)

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
        self.s_encoder= nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.d_encoder = copy.deepcopy(self.s_encoder)
        self.m_encoder = copy.deepcopy(self.s_encoder)
        self.update = nn.Linear(256, 256)
        self.prior_net = PriorNet(512,  # post输入维度
                                  256,  # 潜变量维度
                                  [384])  # 隐藏层维度

        self.recognize_net = RecognizeNet(512,  # post输入维度
                                          256,  # response输入维度
                                          256,  # 潜变量维度
                                          [384])  # 隐藏层维度
        self.atten =nn.MultiheadAttention(embed_dim=256,num_heads=4,dropout=0.2,batch_first=True)
        decoder_layer =  nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.decoder_m  = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.decoder_d = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, dropout=0.2,nhead=4, batch_first=True),num_layers=2)
        self.decoder_s = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, dropout=0.2,nhead=4, batch_first=True),num_layers=2)
        # self.linear = nn.Linear(768, 256)
        self.sigmoid = nn.Sigmoid()
        self.m_linear = nn.Linear(768,voc_size[2])
        self.s_linear = nn.Linear(512,voc_size[0])
        self.d_linear = nn.Linear(512,voc_size[1])
        self.voc_size = voc_size
        self.hcdecoder = Multi_Generator()
        
        
    def sum_embedding(self, embedding):
        return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2) 

    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar): 
        """ 两个高斯分布之间的kl散度公式 """
        kld = -0.5 * torch.mean(
            1 + recog_logvar - prior_logvar - (prior_mu - recog_mu).pow(2).div(torch.exp(prior_logvar))
            - torch.exp(recog_logvar).div(torch.exp(prior_logvar)))
        return kld

    
    def forward(self, input, inference=False,only_pred=False):
        # patient seq representation

        if not inference:
            s_seq = []
            d_seq = []
            m_seq = []
            c_labels = []
            r_labels =[]
            d_labels = []
            c_inps = []
            r_inps = []
            d_inps = []
            d_enc_in = []
            m_enc_in = []
            s_enc_in=[]
            sym_id_labels = []
            med_id_labels = []
            diag_id_labels = []
            loss_multi_target = (torch.ones((len(input), self.voc_size[1])).long()*(-1)).to(self.device)
            for i,input0 in enumerate(input):
                adm, label,_,inp,_,enc_in=input0
                sym = self.sum_embedding(
                    self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))
                )
                dig = self.sum_embedding(
                    self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))
                )
                s_seq.append(sym)
                d_seq.append(dig)
                med = self.sum_embedding(
                    self.dropout(self.embeddings[2](torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)))
                )
                m_seq.append(med)
                sym_id_labels.append(adm[0])
                med_id_labels.append(adm[2])
                diag_id_labels.append(adm[1])
                c_labels.append(torch.LongTensor(label[0]).unsqueeze(dim=0))
                r_labels.append(torch.LongTensor(label[1]).unsqueeze(dim=0))
                d_labels.append(torch.LongTensor(label[2]).unsqueeze(dim=0))
                c_inps.append(torch.LongTensor(inp[0]).unsqueeze(dim=0))
                r_inps.append(torch.LongTensor(inp[1]).unsqueeze(dim=0))
                d_inps.append(torch.LongTensor(inp[2]).unsqueeze(dim=0))
                d_enc_in.append(torch.LongTensor(enc_in[0]).unsqueeze(dim=0))
                m_enc_in.append(torch.LongTensor(enc_in[1]).unsqueeze(dim=0))
                s_enc_in.append(torch.LongTensor(enc_in[2]).unsqueeze(dim=0))
                for j in range(len(adm[1])):
                    loss_multi_target[i,j]=adm[1][j]

            c_labels = torch.cat(c_labels,dim=0)
            r_labels = torch.cat(r_labels,dim=0)
            d_labels = torch.cat(d_labels,dim=0)
            c_inps = torch.cat(c_inps,dim=0)
            r_inps = torch.cat(r_inps,dim=0)
            d_inps = torch.cat(d_inps,dim=0)
            d_enc_in = torch.cat(d_enc_in,dim=0)
            m_enc_in = torch.cat(m_enc_in,dim=0)
            s_enc_in = torch.cat(s_enc_in,dim=0)

            labels = [c_labels,r_labels,d_labels]
            inps = [c_inps,r_inps,d_inps]
            enc_ins = [d_enc_in,m_enc_in,s_enc_in]

            s_seq = torch.cat(s_seq, dim=1)  # (1,seq,dim)
            d_seq = torch.cat(d_seq, dim=1)  # (1,seq,dim)
            m_seq = torch.cat(m_seq, dim=1)  # (1,seq,dim)

            #src_mask,时序信息
            src_mask = torch.triu(torch.full((s_seq.shape[1], s_seq.shape[1]), float('-inf')), diagonal=1).to(self.device)

            s_enc = self.s_encoder(s_seq,mask=src_mask) # (1,seq,dim)
            d_enc = self.d_encoder(d_seq,mask=src_mask) # (1,seq,dim)
            m_enc = self.m_encoder(m_seq,mask=src_mask) # (1,seq,dim)
            s_enc = self.update(s_enc)
            d_enc_new = d_enc + s_enc # (1,seq,dim)
            x = torch.cat([s_enc, d_enc_new], dim=-1) # (1,seq,dim*2)
            _mu, _logvar = self.prior_net(x.squeeze(0)) # (seq,latent)

            mu, logvar = self.recognize_net(x.squeeze(0), m_enc.squeeze(0)) # (1,seq,latent)    
            
            z = self.reparameterise(mu, logvar).unsqueeze(0) # (1,seq,latent)
            z_new,_ = self.atten(z,z,z,need_weights=False,attn_mask=src_mask)# 单向的 atten
            
            if s_seq.shape[1] > 1:
                src_mask_1 = torch.triu(torch.full((s_seq.shape[1]-1, s_seq.shape[1]-1), float('-inf')), diagonal=1).to(self.device)
                sym_pre = self.decoder_s(torch.cat([s_enc[:,:-1,:],z_new[:,:-1,:]],dim=-1),mask=src_mask_1) #(1,seq,dim*3)
                sym_pre = self.s_linear(sym_pre).squeeze(0) #(seq,sym_voc_size)
                sym_truth = torch.zeros_like(sym_pre)
                
                for i in range(len(sym_id_labels)):
                    if i == 0:
                        continue
                    for j in range(len(sym_id_labels[i])):
                        sym_truth[i-1,sym_id_labels[i][j]] = 1.0
                rc_loss_sym = F.binary_cross_entropy_with_logits(sym_pre,sym_truth) #不用bceloss 防溢出

                diag_pre = self.decoder_d(torch.cat([d_enc[:,:-1,:],z_new[:,:-1,:]],dim=-1),mask=src_mask_1) #(1,seq,dim*3)
                diag_pre = self.d_linear(diag_pre).squeeze(0) #(seq,diag_voc_size)
                diag_truth = torch.zeros_like(diag_pre)

                loss_multi_diag = F.multilabel_margin_loss(F.sigmoid(diag_pre), loss_multi_target[1:,:])
                
                for i in range(len(diag_id_labels)):
                    if i == 0:
                        continue
                    for j in range(len(diag_id_labels[i])):
                        diag_truth[i-1,diag_id_labels[i][j]] = 1.0
                rc_loss_diag = F.binary_cross_entropy_with_logits(diag_pre,diag_truth) #不用bceloss 防溢出
            else:
                rc_loss_diag = torch.zeros(1).to(self.device)
                rc_loss_sym = torch.zeros(1).to(self.device)
                loss_multi_diag = torch.zeros(1).to(self.device)
            
            med_pre = self.decoder_m(torch.cat([x, z_new], dim=-1),mask=src_mask) #(1,seq,dim*3)
            med_pre = self.m_linear(med_pre).squeeze(0) #(seq,med_voc_size)
            med_truth = torch.zeros_like(med_pre)
            
            for i in range(len(med_id_labels)):
                for j in range(len(med_id_labels[i])):
                    med_truth[i,med_id_labels[i][j]] = 1.0

            

            rc_loss_med = F.binary_cross_entropy_with_logits(med_pre,med_truth) #不用bceloss 防溢出

            kld_loss = self.gaussian_kld(mu, logvar, _mu, _logvar)

            if only_pred is False:
                chief_loss,his_loss,med_loss = self.hcdecoder(z_new, s_enc, d_enc_new, m_enc,labels,inps,enc_ins) #(1, seq, max_len)
            else:
                chief_loss = his_loss = med_loss = 0
            return rc_loss_sym,rc_loss_diag,rc_loss_med,kld_loss,loss_multi_diag,chief_loss,his_loss,med_loss
        else:
            s_seq = []
            d_seq = []
            for adm,_,_,_,_,_ in input:
                sym = self.sum_embedding(
                    self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))
                )
                dig = self.sum_embedding(
                    self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))
                )
                s_seq.append(sym)
                d_seq.append(dig)

            
            s_seq = torch.cat(s_seq, dim=1)  # (1,seq,dim)
            d_seq = torch.cat(d_seq, dim=1)  # (1,seq,dim)

            #src_mask,时序信息
            src_mask = torch.triu(torch.full((s_seq.shape[1], s_seq.shape[1]), float('-inf')), diagonal=1).to(self.device)

            s_enc = self.s_encoder(s_seq,mask=src_mask)
            d_enc = self.d_encoder(d_seq,mask=src_mask)

            s_enc = self.update(s_enc)
            d_enc_new = d_enc + s_enc # (1,seq,dim)
            
            x = torch.cat([s_enc, d_enc_new],dim=-1)
            _mu, _logvar = self.prior_net(x.squeeze(0)) # (1,seq,latent)
            z = self.reparameterise(_mu, _logvar).unsqueeze(0) # (1,seq,latent)
            z_new,_ = self.atten(z,z,z,need_weights=False,attn_mask=src_mask)
    
            #reconsctrcut med to get med_pre
            med_pre = self.decoder_m(torch.cat([x, z_new], dim=-1),mask=src_mask) #(1,seq,dim*3)
            med_pre = self.sigmoid(self.m_linear(med_pre)).squeeze(0) #(seq,med_voc_size)
            pre_m_seq=[]
            patient_med_pred = []
            patient_diag_pred = []
            patient_sym_pred = []
            
            if s_seq.shape[1] > 1:
                src_mask_1 = torch.triu(torch.full((s_seq.shape[1]-1, s_seq.shape[1]-1), float('-inf')), diagonal=1).to(self.device)
                sym_pre = self.decoder_s(torch.cat([s_enc[:,:-1,:],z_new[:,:-1,:]],dim=-1),mask=src_mask_1) #(1,seq,dim*3)
                sym_pre = self.sigmoid(self.s_linear(sym_pre)).squeeze(0) #(seq,sym_voc_size)

                diag_pre = self.decoder_d(torch.cat([d_enc[:,:-1,:],z_new[:,:-1,:]],dim=-1),mask=src_mask_1) #(1,seq,dim*3)
                diag_pre = self.sigmoid(self.d_linear(diag_pre)).squeeze(0) #(seq,diag_voc_size)
                
            else:
                sym_pre = None
                diag_pre = None
            
            for i in range(med_pre.shape[0]):
                if sym_pre is None or i == 0:
                    patient_sym_pred.append([])
                    patient_diag_pred.append([])
                else:
                    Aadmission_sym_pred=torch.nonzero(sym_pre[i]>0.5).view(-1).tolist() #
                    patient_sym_pred.append(Aadmission_sym_pred)
                    Aadmission_diag_pred=torch.nonzero(diag_pre[i]>0.5).view(-1).tolist() #
                    patient_diag_pred.append(Aadmission_diag_pred)
                

                Aadmission_med_pred=torch.nonzero(med_pre[i]>0.5).view(-1).tolist() #
                patient_med_pred.append(Aadmission_med_pred)

                pre_med = self.sum_embedding(
                    self.dropout(self.embeddings[2](torch.LongTensor(Aadmission_med_pred).unsqueeze(dim=0).to(self.device)))
                )
                pre_m_seq.append(pre_med)
            pre_m_seq = torch.cat(pre_m_seq, dim=1)  # (1,seq,dim)
            med_pre_emb = self.m_encoder(pre_m_seq,mask=src_mask) # (1,seq,dim)   

            
            if only_pred is False:
                seqs,seqs_lens= self.hcdecoder.generate(z_new, s_enc, d_enc_new, med_pre_emb)
                return seqs,seqs_lens
            else:
                return [[patient_sym_pred,sym_pre],[patient_diag_pred,diag_pre],[patient_med_pred,med_pre]]

