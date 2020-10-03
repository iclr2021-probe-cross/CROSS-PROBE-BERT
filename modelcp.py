
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from modeling_bertnewsinglecut import BertModelNew
from modeling_bertnewsinglecut2 import BertModelNew2

from apex import amp
import random


def dot_attention(q, k, v, v_mask=None, dropout=None):
  attention_weights = torch.matmul(q, k.transpose(-1, -2))
  if v_mask is not None:
    attention_weights += v_mask.unsqueeze(1)
  attention_weights = F.softmax(attention_weights, -1)
  output = torch.matmul(attention_weights, v)
  return output




def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)
    #return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    if seq_q.dim() == 1:
        seq_q = seq_q.unsqueeze(0)
    if seq_k.dim() == 1:
        seq_k = seq_k.unsqueeze(0)
    len_q = seq_q.size(1)
    #padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def l1norm(X, dim, eps=1e-5):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-5):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).add(eps).sqrt() + eps
    X = torch.div(X, norm)
    return X





class EncoderCross(nn.Module):
    def __init__(self,opt):
        super(EncoderCross,self).__init__()
        dropout = 0.1
        self.opt = opt
        self.margin = 0.2
    def forward(self, scores, n_img, n_cap, test=False):

        scores = scores.view(n_img,n_cap)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        # keep the maximum violating negative for each query
        eps = 1e-5
        cost_s = cost_s.pow(4).sum(1).add(eps).sqrt().sqrt()#.sqrt()#.div(cost_s.size(1)).mul(2)
        cost_im = cost_im.pow(4).sum(0).add(eps).sqrt().sqrt()#.sqrt()#.div(cost_im.size(0)).mul(2)
        return cost_s.sum() + cost_im.sum()




class EncoderText(nn.Module):

    def __init__(self):
        super(EncoderText, self).__init__()
        self.encoder = BertModelNew.from_pretrained('bert/')
        self.encoder2 = BertModelNew2.from_pretrained('bert/')

        self.fc = nn.Linear(2048,768)
        self.fc2 = nn.Linear(768,30600) 
        self.fc3= nn.Linear(768,1601) 
        self.norm = nn.LayerNorm(768,eps=1e-5)
        self.relu = nn.ReLU()
        self.poly_m = 15#15#5#15
        self.poly_t = 5
        self.poly_code_embeddings = nn.Embedding(self.poly_m, 768)
        self.poly_code_embeddings_t = nn.Embedding(self.poly_t, 768)
        #self.ff = EncoderLayerMinus(768,768,0.1)
    def calPrec(self,pred,grnd):
        idx = grnd!=-1
        pred = pred[idx]
        grnd = grnd[idx]
        corr = pred==grnd
        #prec = corr.sum()/corr.size(0)
        return corr.sum()*1.0,corr.size(0)*1.0#prec


    def getscore(self,text_pos,vision_pos,text_early,context_early):
        head_mask = [None]*20
        cat_early = torch.cat([text_early,context_early],1)
        extended_mask = torch.ones(cat_early.size(0),cat_early.size(1)).long().cuda()
        extended_attention_mask = extended_mask.squeeze()[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        cat_new = self.encoder2.encoder.forward_follow(cat_early.float(),extended_attention_mask, head_mask)
        cat_new = cat_new[0]
        text_late = cat_new[:,:text_early.size(1)].sum(1) + text_pos
        vision_late = cat_new[:,text_early.size(1):].sum(1) + vision_pos
        scores = cosine_similarity(text_late,vision_late)
        return scores

    def forward(self, input_ids, non_pad_mask, vision_feat, vision_mask, gt_labels=None, vision_labels=None, MLM=False, istest=False):
        token_type_ids = torch.zeros_like(non_pad_mask)
        text_output = self.encoder.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids.long().squeeze())
        head_mask = [None]*20
        vision_feat = self.fc(vision_feat)
        vision_feat = self.norm(vision_feat)

        bs = text_output.size(0)
        tl = text_output.size(1)
        vl = vision_feat.size(1)

        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).cuda()
        poly_code_text = torch.arange(self.poly_t, dtype=torch.long).cuda()

        poly_mask = torch.ones(bs,self.poly_m).long().cuda()
        poly_mask_text = torch.ones(bs,self.poly_t).long().cuda()
 
        #poly_code_ids += 1
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(bs, self.poly_m)
        poly_code_text = poly_code_text.unsqueeze(0).expand(bs, self.poly_t)
         
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        poly_codes = self.norm(poly_codes)

        poly_codes_text = self.poly_code_embeddings(poly_code_text)
        poly_codes_text = self.norm(poly_codes_text)


        non_pad_mask_ext = torch.cat([non_pad_mask.squeeze().long(),poly_mask_text],1) 
        extended_attention_mask_text = non_pad_mask_ext.squeeze()[:, None, None, :]
        extended_attention_mask_text = (1.0 - extended_attention_mask_text) * -10000.0

        attention_mask_text = non_pad_mask.squeeze().long()[:, None, None, :]
        attention_mask_text = (1.0 - attention_mask_text) * -10000.0


        extended_vision_mask = torch.cat([vision_mask.long(),poly_mask],1)
        extended_attention_mask_vision = extended_vision_mask.squeeze()[:, None, None, :]
        extended_attention_mask_vision = (1.0 - extended_attention_mask_vision) * -10000.0

        attention_mask_vision = vision_mask.long().squeeze()[:, None, None, :]
        attention_mask_vision = (1.0 - attention_mask_vision) * -10000.0

        text_output = torch.cat([text_output,poly_codes_text],1)
        vision_feat = torch.cat([vision_feat,poly_codes],1)

        textnew = self.encoder2.encoder(text_output,extended_attention_mask_text,head_mask)
        textnew = textnew[0]
        textnew_follow = self.encoder2.encoder.forward_follow(textnew[:,:tl],attention_mask_text,head_mask)
        textnew_follow = textnew_follow[0].sum(1)

        extended_vision_feat = vision_feat
        visionnew = self.encoder2.encoder(extended_vision_feat,extended_attention_mask_vision,head_mask) 
        visionnew = visionnew[0]
        visionnew_follow = self.encoder2.encoder.forward_follow(visionnew[:,:vl],attention_mask_vision,head_mask)
        visionnew_follow = visionnew_follow[0].sum(1)
        if istest:
            vision_early = visionnew[:,vl:]#.view(bs,-1)#dot_attention(poly_codes, visionnew, visionnew)#, poly_vision_mask)
            text_early = textnew[:,tl:]#.view(bs,self.poly_t,-1)#view(bs,1,-1)
            text_pos =  textnew_follow
            vision_pos = visionnew_follow
            return text_pos,vision_pos,text_early,vision_early

        #textnew = textnew[0]
        #visionnew = visionnew[0]
        
        context_vecs = visionnew[:,vl:]#.view(bs,-1)#dot_attention(poly_codes, visionnew, visionnew)#, poly_vision_mask)
        text_out = textnew[:,tl:].view(bs,self.poly_t,-1)#view(bs,1,-1)

        m = self.poly_m
        t = self.poly_t

        if istest == False:
            context_vecs = context_vecs.unsqueeze(0).expand(bs,bs,m,-1).contiguous().view(bs*bs,m,-1)
            text_out = text_out.unsqueeze(1).expand(bs,bs,t,-1).contiguous().view(bs*bs,t,-1)
            poly_mask = poly_mask.unsqueeze(1).expand(bs,bs,m).contiguous().view(bs*bs,m)
            poly_mask_text = poly_mask_text.unsqueeze(1).expand(bs,bs,t).contiguous().view(bs*bs,t)

            textnew_follow = textnew_follow.unsqueeze(1).expand(bs,bs,-1).contiguous().view(bs*bs,-1)
            visionnew_follow = visionnew_follow.unsqueeze(0).expand(bs,bs,-1).contiguous().view(bs*bs,-1)
            

        cat_feat = torch.cat([text_out,context_vecs],1)
        cat_mask = torch.cat([poly_mask_text,  poly_mask],1)
        cat_mask = cat_mask[:, None, None,:]
        cat_mask = (1.0-cat_mask)* -10000.0
        cat_feat = self.encoder2.encoder.forward_follow(cat_feat,cat_mask,head_mask)


        cat_feat = cat_feat[0]
        text_pos = cat_feat[:,:t].sum(1) + textnew_follow 
        vision_pos = cat_feat[:,t:].sum(1) + visionnew_follow

        #else:
        text_output = text_pos
        vision_output = vision_pos

        scores =  cosine_similarity(text_output,vision_output,-1)
        
        margin = 0.2

        scores = scores.view(bs,bs)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
            
        cost_s = (margin + scores - d1).clamp(min=0)
        cost_im = (margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        eps = 1e-5
        cost_s = cost_s.pow(4).sum(1).add(eps).sqrt().sqrt()#.sqrt()#.div(cost_s.size(1))#.sqrt()#.div(cost_s.size(1)).mul(2)
        cost_im = cost_im.pow(4).sum(0).add(eps).sqrt().sqrt()#.sqrt()#.div(cost_im.size(1))#.sqrt()#.div(cost_im.size(0)).mul(2)
        return cost_s.sum().div(cost_s.size(0)) + cost_im.sum().div(cost_s.size(0))


def cosine_similarity(x1, x2, dim=1, eps=1e-5):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()




class CPBERT(object):
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.txt_enc = EncoderText()
        self.cross_att = EncoderCross(opt)#.half()

 
        self.drop = torch.nn.Dropout(p=0.0)

        #if torch.cuda.is_available():
        self.txt_enc.cuda()#.cuda()
        self.cross_att.cuda()#.cuda()
        cudnn.benchmark = True


        params = list(self.txt_enc.parameters())
        params += list(self.cross_att.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.txt_enc, self.optimizer = amp.initialize(self.txt_enc, self.optimizer, opt_level= "O1")
        self.txt_enc = torch.nn.DataParallel(self.txt_enc)
        # Loss and Optimizer
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.txt_enc.load_state_dict(state_dict[0])

    def train_start(self):
        """switch to train mode
        """
        self.txt_enc.train()
        self.cross_att.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.txt_enc.eval()
        self.cross_att.eval()

    def forward_emb(self, images, captions,  target_mask, vision_mask, volatile=False, istest = False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images.float(), volatile=volatile)
        captions = torch.LongTensor(captions)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()#.cuda()
            captions = captions.cuda()#.cuda()
        # Forward

        n_img = images.size(0)
        n_cap = captions.size(0)
        if istest:
            images = images.unsqueeze(1).expand(n_img,n_cap,images.size(1),images.size(2)).contiguous().view(-1,images.size(1),images.size(2))
            captions = captions.unsqueeze(0).expand(n_img,n_cap,captions.size(1)).contiguous().view(-1,captions.size(1))

        attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        token_type_ids = torch.zeros_like(attention_mask)

        video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()
        if istest:
            video_non_pad_mask = video_non_pad_mask.unsqueeze(1).expand(n_img,n_cap,images.size(1)).contiguous().view(-1,images.size(1))


        scores = self.txt_enc(captions, attention_mask,images,video_non_pad_mask,istest)
        return scores

    def forward_embtest(self, images,  captions, target_mask, vision_mask):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images.float())
        captions = torch.LongTensor(captions)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()#.cuda()
            captions = captions.cuda()#.cuda()
        # Forward
        n_img = images.size(0)
        n_cap = captions.size(0)

        attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        img_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()
        text_pos,vision_pos,text_el,vision_el = self.txt_enc(captions,attention_mask,images,img_non_pad_mask, istest = True)
        return text_pos,vision_pos,text_el,vision_el 

 
    def forward_loss(self, img_emb,cap_emb, cap_len, text_non_pad_mask, text_slf_attn_mask,img_non_pad_mask,img_slf_attn_mask, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        scores = self.cross_att(img_emb,cap_emb,cap_len,text_non_pad_mask, text_slf_attn_mask,img_non_pad_mask,img_slf_attn_mask)
        loss = self.criterion(scores)
        self.logger.update('Le', loss.item(), scores.size(0))
        return loss

    def train_emb(self, images, captions, target_mask, vision_mask, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
         
        # measure accuracy and record loss
        scores = self.forward_emb(images, captions, target_mask, vision_mask)
        # measure accuracy and record loss

        self.optimizer.zero_grad()
        if scores is not None:
           loss = scores.sum()
        else:
           return
        # compute gradient and do SGD step
        #loss.backward()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
           scaled_loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()



