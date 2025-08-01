import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature=nn.parameter.Parameter(torch.tensor(10.), requires_grad=True)
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.fp16 = config['half']
        self.validate = False

    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW)) -2
        WW = W ** 2
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            data.update({
                'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                'mkpts0_f': data['all_mkpts0_c'][:len(data['mconf']),...],
                'mkpts1_f': data['all_mkpts1_c'][:len(data['mconf']),...],
                'all_mkpts0_f': data['all_mkpts0_c'],
                'all_mkpts1_f': data['all_mkpts1_c'],
            })
            return

        # compute pixel-level confidence matrix
        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            feat_f0, feat_f1 = feat_0[...,:-self.local_regress_slicedim], feat_1[...,:-self.local_regress_slicedim]
            feat_ff0, feat_ff1 = feat_0[...,-self.local_regress_slicedim:], feat_1[...,-self.local_regress_slicedim:]
            feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5
            conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
           

        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.W+2, self.W+2, self.W+2, self.W+2)
        softmax_matrix_f = softmax_matrix_f[...,1:-1,1:-1,1:-1,1:-1].reshape(M, self.WW, self.WW)

        # for fine-level supervision
        if self.training or self.validate:
            data.update({'conf_matrix_f': softmax_matrix_f})

        # compute pixel-level absolute kpt coords
        self.get_fine_ds_match(softmax_matrix_f, data)

        # generate seconde-stage 3x3 grid
        idx_l, idx_r = data['idx_l'], data['idx_r']
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
        ## our aim is to obtain all matches
        # m_ids = m_ids[:len(data['mconf'])]

        idx_l_iids, idx_l_jids = idx_l // W, idx_l % W
        idx_r_iids, idx_r_jids = idx_r // W, idx_r % W
        m_ids, idx_l_iids, idx_l_jids, idx_r_iids, idx_r_jids = m_ids.reshape(-1), idx_l_iids.reshape(-1), idx_l_jids.reshape(-1), idx_r_iids.reshape(-1), idx_r_jids.reshape(-1)

        delta = create_meshgrid(3, 3, True, softmax_matrix_f.device).to(torch.long) # [1, 3, 3, 2]
       
        m_ids = m_ids[...,None,None].expand(-1, 3, 3)

        idx_l_iids = idx_l_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_l_jids = idx_l_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]

        idx_r_iids = idx_r_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]

        if idx_l.numel() == 0:
            data.update({
                'mkpts0_f': data['all_mkpts0_c'][:len(data['mconf']),...],
                'mkpts1_f': data['all_mkpts1_c'][:len(data['mconf']),...],
                'all_mkpts0_f': data['all_mkpts0_c'],
                'all_mkpts1_f': data['all_mkpts1_c'],
            })
            return

        # compute second-stage heatmap
        feat_ff0 = feat_ff0.reshape(M, self.W+2, self.W+2, self.local_regress_slicedim)
        feat_ff1 = feat_ff1.reshape(M, self.W+2, self.W+2, self.local_regress_slicedim)
        feat_ff0 = feat_ff0[m_ids, idx_l_iids, idx_l_jids].view(-1, 9, self.local_regress_slicedim)
        feat_ff1 = feat_ff1[m_ids, idx_r_iids, idx_r_jids].view(-1, 9, self.local_regress_slicedim)

        feat_ff0_picked = feat_ff0_picked = feat_ff0[:, 9//2, :]
        feat_ff1_picked = feat_ff1_picked = feat_ff1[:, 9//2, :]
        avg_feat_ff_picked = (feat_ff0_picked + feat_ff1_picked) / 2.

        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            conf_matrix_ff0 = torch.einsum('mc,mrc->mr', avg_feat_ff_picked, feat_ff0 /(self.local_regress_slicedim)**.5)
            conf_matrix_ff1 = torch.einsum('mc,mrc->mr', avg_feat_ff_picked, feat_ff1 /(self.local_regress_slicedim)**.5)
        conf_matrix_ff0 = conf_matrix_ff0.reshape(-1, 9)
        conf_matrix_ff1 = conf_matrix_ff1.reshape(-1, 9)
        conf_matrix_ff0 = F.softmax(conf_matrix_ff0 / self.local_regress_temperature, -1)
        conf_matrix_ff1 = F.softmax(conf_matrix_ff1 / self.local_regress_temperature, -1)
        heatmap0 = conf_matrix_ff0.reshape(-1, 3, 3)
        heatmap1 = conf_matrix_ff1.reshape(-1, 3, 3)

        # compute coordinates from heatmap
        coords_normalized0 = dsnt.spatial_expectation2d(heatmap0[None], True)[0]
        coords_normalized1 = dsnt.spatial_expectation2d(heatmap1[None], True)[0]
        
        if data['bs'] == 1:
            scale0 = scale * data['scale0'] if 'scale0' in data else scale
            scale1 = scale * data['scale1'] if 'scale0' in data else scale
        else:
            scale0 = scale * data['scale0'][data['b_ids']][:,None,:].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale
            scale1 = scale * data['scale1'][data['b_ids']][:,None,:].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale

       
        # compute subpixel-level absolute kpt coords
        self.get_fine_match_local(coords_normalized0, coords_normalized1, data, scale0, scale1)

    def get_fine_match_local(self, coords_normed0, coords_normed1, data, scale0, scale1):
        all_mkpts0_c, all_mkpts1_c = data['all_mkpts0_c'], data['all_mkpts1_c']
        
        # mkpts0_f and mkpts1_f
        all_mkpts0_f = all_mkpts0_c + (coords_normed0 * (3 // 2) * scale0)
        all_mkpts1_f = all_mkpts1_c + (coords_normed1 * (3 // 2) * scale1)

        data.update({
            "mkpts0_f": all_mkpts0_f[:len(data['mconf']),...],
            "mkpts1_f": all_mkpts1_f[:len(data['mconf']),...],
            "all_mkpts0_f": all_mkpts0_f,
            "all_mkpts1_f": all_mkpts1_f
        })

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = conf_matrix.shape

        conf_matrix = conf_matrix.reshape(m, -1)
        _, idx = torch.max(conf_matrix, dim = -1)
        idx = idx[:,None]
        idx_l, idx_r = idx // WW, idx % WW

        data.update({'idx_l': idx_l, 'idx_r': idx_r})

        if self.fp16:
            grid = create_meshgrid(W, W, False, conf_matrix.device, dtype=torch.float16) - W // 2 + 0.5 # kornia >= 0.5.1
        else:
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale

        if torch.is_tensor(scale0) and scale0.numel() > 1: # scale0 is a tensor
            all_mkpts0_f = (data['all_mkpts0_c'][:,None,:] + (delta_l * scale0[:,None,:])).reshape(-1, 2)
            all_mkpts1_f = (data['all_mkpts1_c'][:,None,:] + (delta_r * scale1[:,None,:])).reshape(-1, 2)
        else: # scale0 is a float
            all_mkpts0_f = (data['all_mkpts0_c'][:,None,:] + (delta_l * scale0)).reshape(-1, 2)
            all_mkpts1_f = (data['all_mkpts1_c'][:,None,:] + (delta_r * scale1)).reshape(-1, 2)
        

        data.update({
            "mkpts0_c": all_mkpts0_f[:len(data['mconf']),...],
            "mkpts1_c": all_mkpts1_f[:len(data['mconf']),...],
            "all_mkpts0_c": all_mkpts0_f,
            "all_mkpts1_c": all_mkpts1_f
        })


        