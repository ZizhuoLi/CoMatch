import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import Attention, crop_feature, pad_feature
from einops.einops import rearrange
from collections import OrderedDict
from ..utils.position_encoding import RoPEPositionEncodingSine
import numpy as np
from loguru import logger    

class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

       
        self.mlp=nn.Sequential(
                     nn.Conv2d(d_model*2, d_model, kernel_size=1, bias=False),
                     nn.ReLU(True),
                     nn.Conv2d(d_model, d_model, kernel_size=3,padding=1, bias=False),
                )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, source, x_mask=None, source_mask=None, x_matchability_score=None, source_matchability_score=None, name=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        
        if source_matchability_score == None:
            query, source = self.norm1(self.aggregate(x).permute(0,2,3,1)), self.norm1(self.max_pool(source).permute(0,2,3,1)) # [N, H, W, C]
        else:
            pooled_source_matchability_score = self.max_pool(source_matchability_score).permute(0,2,3,1) # [N,1 H, W]->[N,H, W, 1]
            
            source_matchability_score_unfold =  F.unfold(source_matchability_score, kernel_size=(self.agg_size1, self.agg_size1), stride=self.agg_size1)
            source_matchability_score_unfold = torch.softmax(source_matchability_score_unfold, dim=1) # [N, ww, L]

            source_unfold = F.unfold(source, kernel_size=(self.agg_size1, self.agg_size1), stride=self.agg_size1) # [N, wwC, L]
            source_unfold = rearrange(source_unfold, 'n (c ww) l -> n c ww l', ww=self.agg_size1**2) # # [N, C, ww, L]
            source_unfold = torch.sum(source_unfold * source_matchability_score_unfold.unsqueeze(1), dim=2) # [N, C, L]
            weighted_source = rearrange(source_unfold, 'n c (h w) -> n c h w', h=H1//self.agg_size1, w=W1//self.agg_size1)
            query, source = self.norm1(self.aggregate(x * x_matchability_score).permute(0,2,3,1)), self.norm1(weighted_source.permute(0,2,3,1)) # [N, H, W, C]

        if x_mask is not None:
            x_mask, source_mask = map(lambda x: self.max_pool(x.float()).bool(), [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        
        if source_matchability_score != None:
            value = value * pooled_source_matchability_score

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)

      
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1)).permute(0, 2, 3, 1) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        
        self.full_config = config
        self.fp32 = not (config['mp'] or config['half'])
        config = config['coarse']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.agg_size0, self.agg_size1 = config['agg_size0'], config['agg_size1']
        self.rope = config['rope']

        self_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], config['rope'], config['npe'], self.fp32)
        cross_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], False, config['npe'], self.fp32)
        self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' else copy.deepcopy(cross_layer) for _ in self.layer_names])
        
        self.matchability_predictor = nn.ModuleList([nn.Sequential(
            nn.Conv2d(config['d_model'], config['d_model'], kernel_size=3, padding=1, bias=False, groups=config['d_model']),
            nn.ReLU(inplace=True),
            nn.Conv2d(config['d_model'], 1, kernel_size=1, stride=1, bias=True)) for _ in range(len( self.layer_names)//2 - 1)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]

        feature_cropped = False
        if bs == 1 and mask0 is not None and mask1 is not None:
            mask_H0, mask_W0, mask_H1, mask_W1 = mask0.size(-2), mask0.size(-1), mask1.size(-2), mask1.size(-1)
            mask_h0, mask_w0, mask_h1, mask_w1 = mask0[0].sum(-2)[0], mask0[0].sum(-1)[0], mask1[0].sum(-2)[0], mask1[0].sum(-1)[0]
            mask_h0, mask_w0, mask_h1, mask_w1 = mask_h0//self.agg_size0*self.agg_size0, mask_w0//self.agg_size0*self.agg_size0, mask_h1//self.agg_size1*self.agg_size1, mask_w1//self.agg_size1*self.agg_size1
            feat0 = feat0[:, :, :mask_h0, :mask_w0]
            feat1 = feat1[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        
        matchability_score_list0, matchability_score_list1 = [], []


        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if feature_cropped:
                mask0, mask1 = None, None
            if name == 'self':
                if i == 0:
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
                else:
                    feat0 = layer(feat0, feat0, mask0, mask0, matchability_score0, matchability_score0,name)
                    feat1 = layer(feat1, feat1, mask1, mask1, matchability_score1, matchability_score1,name)
            elif name == 'cross':
                if i == 1:
                    feat0 = layer(feat0, feat1, mask0, mask1)
                    feat1 = layer(feat1, feat0, mask1, mask0)
                    matchability_score0 = torch.sigmoid(self.matchability_predictor[i//2](feat0))
                    matchability_score1 = torch.sigmoid(self.matchability_predictor[i//2](feat1))
                    
                else:
                    feat0 = layer(feat0, feat1, mask0, mask1, matchability_score0, matchability_score1,name)
                    feat1 = layer(feat1, feat0, mask1, mask0, matchability_score1, matchability_score0,name)


                    if feature_cropped:
                        # padding feature
                        bs, c, mask_h0, mask_w0 = matchability_score0.size()
                        if mask_h0 != mask_H0:
                            matchability_score0 = torch.cat([matchability_score0, torch.zeros(bs, c, mask_H0-mask_h0, mask_W0, device=matchability_score0.device, dtype=matchability_score0.dtype)], dim=-2)
                        elif mask_w0 != mask_W0:
                            matchability_score0 = torch.cat([matchability_score0, torch.zeros(bs, c, mask_H0, mask_W0-mask_w0, device=matchability_score0.device, dtype=matchability_score0.dtype)], dim=-1)

                        bs, c, mask_h1, mask_w1 = matchability_score1.size()
                        if mask_h1 != mask_H1:
                            matchability_score1 = torch.cat([matchability_score1, torch.zeros(bs, c, mask_H1-mask_h1, mask_W1, device=matchability_score1.device, dtype=matchability_score1.dtype)], dim=-2)
                        elif mask_w1 != mask_W1:
                            matchability_score1 = torch.cat([matchability_score1, torch.zeros(bs, c, mask_H1, mask_W1-mask_w1, device=matchability_score1.device, dtype=matchability_score1.dtype)], dim=-1)
                    matchability_score_list0.append(matchability_score0)
                    matchability_score_list1.append(matchability_score1)
                    
                    if i != 7:
                        matchability_score0 = torch.sigmoid(self.matchability_predictor[i//2](feat0))
                        matchability_score1 = torch.sigmoid(self.matchability_predictor[i//2](feat1))

            else:
                raise KeyError

        if feature_cropped:
            # padding feature
            bs, c, mask_h0, mask_w0 = feat0.size()
            if mask_h0 != mask_H0:
                feat0 = torch.cat([feat0, torch.zeros(bs, c, mask_H0-mask_h0, mask_W0, device=feat0.device, dtype=feat0.dtype)], dim=-2)
            elif mask_w0 != mask_W0:
                feat0 = torch.cat([feat0, torch.zeros(bs, c, mask_H0, mask_W0-mask_w0, device=feat0.device, dtype=feat0.dtype)], dim=-1)

            bs, c, mask_h1, mask_w1 = feat1.size()
            if mask_h1 != mask_H1:
                feat1 = torch.cat([feat1, torch.zeros(bs, c, mask_H1-mask_h1, mask_W1, device=feat1.device, dtype=feat1.dtype)], dim=-2)
            elif mask_w1 != mask_W1:
                feat1 = torch.cat([feat1, torch.zeros(bs, c, mask_H1, mask_W1-mask_w1, device=feat1.device, dtype=feat1.dtype)], dim=-1)
        return feat0, feat1, matchability_score_list0, matchability_score_list1
    
class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = FullAttention() 
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message
    
class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
       

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        
        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()
    
class LocalFeatureTransformer_loftr(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer_loftr, self).__init__()

        self.config = config
        self.d_model = config['d_model'] 
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

       
        for layer, name in zip(self.layers, self.layer_names):
           
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
    
