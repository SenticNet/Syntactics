import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
from copy import deepcopy

from fastNLP.core.const import Const as C
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules.encoder import MultiHeadAttention
from fastNLP.modules.decoder import ConditionalRandomField
from fastNLP.modules import decoder, encoder
from fastNLP.modules.decoder.crf import allowed_transitions
from .transformer import MultiHeadAttn, TransformerEncoder, SinusoidalPositionalEmbedding, LearnedPositionalEmbedding, TransformerLayer
from .utils import initial_parameter

from flair.tokenization import SpacyTokenizer
from flair.data import Sentence

class MultitaskTagger(nn.Module):
    def __init__(self, embed, p_num, c_num, cl_size=4, loss_weight=0.9,
                 num_layers=2, n_head=4, head_dims=128, dropout=0.3, fc_dropout=0.3, c_vocab=None):
        super().__init__()
        self.on_np = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_layers = num_layers
        self.embed = embed
        embed_size = self.embed.embedding_length
        d_model = n_head * head_dims
        feedforward_dim = int(2 * d_model)
        after_norm = True
        self.cl_size = cl_size
        self.loss_weight = loss_weight
        self.cl_criterion = self.cl_criterion_min
        
        self.in_fc = nn.Linear(embed_size, d_model)

        self.encoder_p = TransformerEncoder(1, d_model, n_head, feedforward_dim, dropout, after_norm, attn_type='transformer', scale=False, dropout_attn=None, pos_embed='sin')
        self.p_trans_dropout = nn.Dropout(fc_dropout)
        self_attn_p = MultiHeadAttn(d_model, n_head, dropout)
        self.layers_p = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn_p), feedforward_dim, after_norm, dropout) for _ in range(num_layers)])
        self.bridge_p = nn.ModuleList([CNNBiaffine(d_model) for _ in range(num_layers)])
        
        self.p_dropout = nn.Dropout(fc_dropout)
        self.hid2p = nn.Linear(d_model, p_num)
        self.p_crf = ConditionalRandomField(p_num, include_start_end_trans=False)


        self.encoder_c = TransformerEncoder(1, d_model, n_head, feedforward_dim, dropout, after_norm, attn_type='transformer', scale=False, dropout_attn=None, pos_embed='sin')
        self.c_trans_dropout = nn.Dropout(fc_dropout)
        self_attn_c = MultiHeadAttn(d_model, n_head, dropout)
        self.layers_c = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn_c), feedforward_dim, after_norm, dropout) for _ in range(num_layers)])
        self.bridge_c = nn.ModuleList([CNNBiaffine(d_model) for _ in range(num_layers)])

        self.c_dropout = nn.Dropout(fc_dropout)
        self.hid2c = nn.Linear(d_model, c_num)
        trans = None
        if c_vocab is not None:
            assert len(c_vocab)==c_num, "The number of classes should be same with the length of target vocabulary."
            trans = allowed_transitions(c_vocab, include_start_end=True) 
        self.c_crf = ConditionalRandomField(c_num, include_start_end_trans=True, allowed_transitions=trans)   
        
        

    def _forward(self, words, seq_len=None, target=None, c=None, on_np=False, cl_samples=None):
        
        sentences = []
        for s in words:
            if not isinstance(s, list):
                s = [str(w) for w in s]
            s = Sentence(s)
            s.to(self.device)
            sentences.append(s)
        self.embed.embed(sentences)
        words, mask = self._make_padded_tensor_for_batch(sentences, None)
        
        words = self.in_fc(words)
        
        p_hid = self.encoder_p(words, mask)
        p_hid = self.p_trans_dropout(p_hid)
        
        if on_np:
            cl_words = []
            for s in cl_samples['words']:
                if not isinstance(s, list):
                    s = [str(w) for w in s]
                s = Sentence(s)
                s.to(self.device)
                cl_words.append(s)
            cl_c = cl_samples['c'].to(self.device)
            bsz = cl_c.size()[0]//self.cl_size
            cl_c = torch.chunk(cl_c,bsz,0)
            self.embed.embed(cl_words)
            np_max_len = mask.size()[1]
            cl_words, cl_mask = self._make_padded_tensor_for_batch(cl_words, np_max_len)

            cl_words = self.in_fc(cl_words)
            cl_hid = self.encoder_c(cl_words, cl_mask) # (cl_size*bz) * seq_len * d_model
            cl_hid = self.c_trans_dropout(cl_hid)
            cl_hid = torch.chunk(cl_hid,bsz,0)
            cl_mask = torch.chunk(cl_mask,bsz,0)
            c_feats = []
            c_mask = []
            c_target = []
            for idx, hid in enumerate(cl_hid):
                # hid: cl_size * seq_len * d_model
                # loop for bsz
                cl_idx, cl_feats = self.cl_criterion(p_hid[idx], hid)
                hid = hid.detach()
                c_feats.append(cl_feats)
                c_mask.append(cl_mask[idx][cl_idx])
                c_target.append(cl_c[idx][cl_idx])
            c_feats = torch.stack(c_feats)
            c_mask = torch.stack(c_mask)
            c_target = torch.stack(c_target)
        else:
            # training on conll or predicting
            c_feats = self.encoder_c(words, mask)
            c_mask = mask
            
        p_feats = p_hid
        c_feats = self.c_trans_dropout(c_feats)
        for i in range(self.num_layers):
            c_feats_n = self.layers_c[i](c_feats, c_mask)
            c_feats_n = self.bridge_c[i](c_feats_n, p_feats.detach())
            p_feats = self.layers_p[i](p_feats, mask)
            p_feats = self.bridge_p[i](p_feats, c_feats)
            c_feats = c_feats_n
                
        c_feats = self.hid2c(c_feats)
        c_feats = self.c_dropout(c_feats)
        c_feats = F.log_softmax(c_feats, dim=-1)
        
        p_feats = self.hid2p(p_feats)
        p_feats = self.p_dropout(p_feats)
        p_feats = F.log_softmax(p_feats, dim=-1)
        
            
        if target is None and c is None:
            # predict
            p_paths, _ = self.p_crf.viterbi_decode(p_feats, mask) 
            c_paths, _ = self.c_crf.viterbi_decode(c_feats, mask) 
            return {'pred': p_paths, 'c_pred': c_paths}
        else:
            p_loss = self.p_crf(p_feats, target, mask).mean()
            if on_np:
                c_loss = self.c_crf(c_feats, c_target, c_mask).mean() 
                loss =  self.loss_weight * p_loss + (1-self.loss_weight) * c_loss
                return {C.LOSS:loss, 'p_loss': p_loss ,'c_loss': c_loss}
            else:
                c_loss = self.c_crf(c_feats, c, mask).mean() 
                loss = 0.4 * p_loss + 0.6 * c_loss
                return {C.LOSS:loss, 'p_loss': p_loss ,'c_loss': c_loss}

    def forward(self, words, seq_len, target, c=None, on_np=False, cl_samples=None):
        if self.on_np:
            on_np = True
        return self._forward(words, seq_len, target, c, on_np=on_np, cl_samples=cl_samples)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len, target=None, c=None, on_np=False, cl_samples=None)
    
    def unfreeze(self, modules, unFreeze):
        for module in modules:
            for param in module.parameters():
                param.requires_grad=unFreeze

    
    def cl_criterion_min(self, p, c):
        cos = nn.CrossEntropyLoss()
        criterion = 1
        min_idx = 0
        for idx, hid in enumerate(c):
            cs = cos(p, hid)
            if cs < criterion:
                criterion = cs
                min_idx = idx
        return min_idx, c[min_idx]
    
    def _make_padded_tensor_for_batch(self, sentences, CL_max_len):
        names = self.embed.get_names()
        lengths = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch = max(lengths)
        if CL_max_len is not None: 
            longest_token_sequence_in_batch = CL_max_len
        pre_allocated_zero_tensor = torch.zeros(
            self.embed.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=self.device,
        )
        all_embs = list()
        # mask = list()
        mask = torch.ones([len(sentences), longest_token_sequence_in_batch], device=self.device)
        
        for idx, sentence in enumerate(sentences):
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                mask[idx][-nb_padding_tokens:]-=1
                all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
                t = pre_allocated_zero_tensor[: self.embed.embedding_length * nb_padding_tokens]
                all_embs.append(t)
            elif nb_padding_tokens == 0:
                all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
            elif nb_padding_tokens < 0:
                sen_emb = []
                for idx, token in enumerate(sentence):
                    if idx >= longest_token_sequence_in_batch:
                        break
                    for emb in token.get_each_embedding(names):
                        sen_emb.append(emb)
                all_embs += sen_emb

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embed.embedding_length,
            ]
        )

        return sentence_tensor, mask

    
class CNNBiaffine(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        kernel_sizes = [7, 5, 3]
        filter_nums = [40, 30, 20]
        self.cnn = nn.ModuleList([nn.Conv1d(
            d_model, filter_nums[i], kernel_size=kernel_sizes[i], bias=True,
            padding=kernel_sizes[i] // 2)
            for i in range(len(kernel_sizes))])
        input_dim = 0 #d_model
        for i in filter_nums:
            input_dim += i
        
        self.U = nn.Parameter(torch.Tensor(d_model, d_model), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(d_model), requires_grad=True)
        initial_parameter(self)
        
        self.dropout_layer = nn.Dropout(0.1)
        
        self.fc = nn.Linear(input_dim, d_model)
        self.fcx = nn.Linear(d_model*2, d_model)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x1, x2):
        x = x2.permute(0, 2, 1)
        x = [conv(x).permute(0, 2, 1) for conv in self.cnn]
        x = torch.cat(x, dim=-1)
        x = F.relu(x)
        x = self.fc(x)
        x = torch.tanh(x)
        
        attn = x1.matmul(self.U)
        attn = attn.bmm(x2.transpose(-1, -2))
        attn = attn + x2.matmul(self.bias).unsqueeze(1) 
        attn = F.softmax(attn, dim=-1)  
        attn = self.dropout_layer(attn)
        output = torch.bmm(attn, x2)
        output = torch.tanh(output)
        
        x = x + output
        x1 = torch.cat((x1,x),-1)
        x1 = self.fcx(x1)
        x1 = self.dropout(x1)
        
        return x1
    

class BiaffineAttn(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        self.U = nn.Parameter(torch.Tensor(d_model, d_model), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(d_model), requires_grad=True)
        initial_parameter(self)
        
        self.fc = nn.Linear(d_model, d_model//2)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x1, x2):
        
        attn = x2.matmul(self.U)
        attn = attn.bmm(x1.transpose(-1, -2))
        attn = attn + x1.matmul(self.bias).unsqueeze(1) # batch_size x max_len x max_len
        attn = F.softmax(attn, dim=-1)  
        attn = self.dropout_layer(attn)
        output = torch.bmm(attn, x1) # batch_size x max_len x d_model
        output = self.fc(output)
        output = F.relu(output)
        
        return output


