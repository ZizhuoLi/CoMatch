from math import log
from loguru import logger as loguru_logger

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from kornia.utils import create_meshgrid
from src.utils.plotting import make_matching_figures
from src.utils.metrics import symmetric_epipolar_distance
from .geometry import warp_kpts, pose2essential_fundamental, warp_kpts_ada

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # # added by lzz
    # scale_f = config['LOFTR']['RESOLUTION'][1] * 2
    # h0_f, w0_f, h1_f, w1_f = map(lambda x: x // scale_f, [H0, W0, H1, W1])
    # (
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     depth_mask0_f,
    #     depth_mask1_f,
    #     _,
    #     _,
    #     _,
    #     _,
    # ) = get_scale_gt_matrix5(
    #     data,
    #     scale_f,
    #     N,
    #     H0,
    #     W0,
    #     H1,
    #     W1,
    #     device,
    #     require_depth_mask=True,
    #     obj_geod_th=1e-4,
    # )

    # depth_mask0_f = rearrange(depth_mask0_f, "n (h w) -> n h w", h=h0_f, w=w0_f)
    # depth_mask1_f = rearrange(depth_mask1_f, "n (h w) -> n h w", h=h1_f, w=w1_f)

    # depth_mask0_c = (
    #     rearrange(
    #         depth_mask0_f, "n (s_h h) (s_w w) -> n (h w) s_h s_w ", s_h=h0, s_w=w0
    #     ).sum(dim=1, keepdim=True)
    #     > 0
    # )  # / (scale_l1l2**2) > 0.1
    # depth_mask1_c = (
    #     rearrange(
    #         depth_mask1_f, "n (s_h h) (s_w w) -> n (h w) s_h s_w ", s_h=h1, s_w=w1
    #     ).sum(dim=1, keepdim=True)
    #     > 0
    # )  # / (scale_l1l2**2) > 0.1

    # depth_mask0_c = depth_mask0_c.float()
    # depth_mask1_c = depth_mask1_c.float()
    # data.update(
    #     {
    #         "spv_matchability_map0": depth_mask0_c,  # [N, 1, h0, w0]
    #         "spv_matchability_map1": depth_mask1_c,  # [N, 1, h1, w1]
    #     }
    # )

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)

    # _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    # _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    valid_mask0, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    valid_mask1, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])

    matchability_map0 = rearrange(valid_mask0, "n (h w) -> n h w", h=h0, w=w0).unsqueeze(1)
    matchability_map1 = rearrange(valid_mask1, "n (h w) -> n h w", h=h1, w=w1).unsqueeze(1)
    data.update(
        {
            "spv_matchability_map0": matchability_map0,  # [N, 1, h0, w0]
            "spv_matchability_map1": matchability_map1,  # [N, 1, h1, w1]
        }
    )
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round()
    # calculate the overlap area between warped patch and grid patch as the loss weight.
    # (larger overlap area between warped patches and grid patch with higher weight)
    # (overlap area range from [0, 1] rather than [0.25, 1] as the penalty of warped kpts fall on midpoint of two grid kpts)
    if config.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT:
        w_pt0_c_error = (1.0 - 2*torch.abs(w_pt0_c - w_pt0_c_round)).prod(-1)
    w_pt0_c_round = w_pt0_c_round[:, :, :].long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1

    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # use overlap area as loss weight
    if config.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT:
        conf_matrix_error_gt = w_pt0_c_error[b_ids, i_ids]  # weight range: [0.0, 1.0]
        data.update({'conf_matrix_error_gt': conf_matrix_error_gt})


    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        loguru_logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i,
        # 'spv_w_pt0_i_c': w_pt0_c_round,
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

@static_vars(counter = 0)
@torch.no_grad()
def spvs_fine(data, config, logger = None):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2], used as subpixel-level gt
            "conf_matrix_f_gt": [M, WW, WW], M is the number of all coarse-level gt matches
            "conf_matrix_f_error_gt": [Mp], Mp is the number of all pixel-level gt matches
            "m_ids_f": [Mp]
            "i_ids_f": [Mp]
            "j_ids_f_di": [Mp]
            "j_ids_f_dj": [Mp]
            }
    """
    # 1. misc
    pt1_i = data['spv_pt1_i']
    W = config['LOFTR']['FINE_WINDOW_SIZE']
    WW = W*W
    scale = config['LOFTR']['RESOLUTION'][1]
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    hf0, wf0, hf1, wf1 = data['hw0_f'][0], data['hw0_f'][1], data['hw1_f'][0], data['hw1_f'][1]  # h, w of fine feature
    assert not config.LOFTR.ALIGN_CORNER, 'only support training with align_corner=False for now.'

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
    scalei0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
    scalei1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale

    # 3. compute gt
    m = b_ids.shape[0]
    if m == 0:  # special case: there is no coarse gt
        conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device)

        data.update({'conf_matrix_f_gt': conf_matrix_f_gt})
        if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
            conf_matrix_f_error_gt = torch.zeros(1, device=device)
            data.update({'conf_matrix_f_error_gt': conf_matrix_f_error_gt})
        
        data.update({'expec_f': torch.zeros(1, 2, device=device)})
        data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
    else:
        grid_pt0_f = create_meshgrid(hf0, wf0, False, device) - W // 2 + 0.5 # [1, hf0, wf0, 2] # use fine coordinates
        grid_pt0_f = rearrange(grid_pt0_f, 'n h w c -> n c h w')
        # 1. unfold(crop) all local windows
        if config.LOFTR.ALIGN_CORNER is False: # even windows
            assert W==8
            grid_pt0_f_unfold = F.unfold(grid_pt0_f, kernel_size=(W, W), stride=W, padding=0)
        grid_pt0_f_unfold = rearrange(grid_pt0_f_unfold, 'n (c ww) l -> n l ww c', ww=W**2) # [1, hc0*wc0, W*W, 2]
        grid_pt0_f_unfold = repeat(grid_pt0_f_unfold[0], 'l ww c -> N l ww c', N=N)

        # 2. select only the predicted matches
        grid_pt0_f_unfold = grid_pt0_f_unfold[data['b_ids'], data['i_ids']]  # [m, ww, 2]
        grid_pt0_f_unfold = scalei0[:,None,:] * grid_pt0_f_unfold  # [m, ww, 2]
        
        # 3. warp grids and get covisible & depth_consistent mask
        correct_0to1_f = torch.zeros(m, WW, device=device, dtype=torch.bool)
        w_pt0_i = torch.zeros(m, WW, 2, device=device, dtype=torch.float32)
        for b in range(N):
            mask = b_ids == b  # mask of each batch
            match = int(mask.sum())
            correct_0to1_f_mask, w_pt0_i_mask = warp_kpts(grid_pt0_f_unfold[mask].reshape(1,-1,2), data['depth0'][[b],...],
                    data['depth1'][[b],...], data['T_0to1'][[b],...], 
                    data['K0'][[b],...], data['K1'][[b],...]) # [k, WW], [k, WW, 2]
            correct_0to1_f[mask] = correct_0to1_f_mask.reshape(match, WW)
            w_pt0_i[mask] = w_pt0_i_mask.reshape(match, WW, 2)
        
        # 4. calculate the gt index of pixel-level refinement
        delta_w_pt0_i = w_pt0_i - pt1_i[b_ids, j_ids][:,None,:] # [m, WW, 2]
        del b_ids, i_ids, j_ids
        delta_w_pt0_f = delta_w_pt0_i / scalei1[:,None,:] + W // 2 - 0.5
        delta_w_pt0_f_round = delta_w_pt0_f[:, :, :].round()
        if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
            # calculate the overlap area between warped patch and grid patch as the loss weight.
            w_pt0_f_error = (1.0 - 2*torch.abs(delta_w_pt0_f - delta_w_pt0_f_round)).prod(-1) # [0, 1]     
        delta_w_pt0_f_round = delta_w_pt0_f_round.long()
    
        nearest_index1 = delta_w_pt0_f_round[..., 0] + delta_w_pt0_f_round[..., 1] * W # [m, WW]
        
        # corner case: out of fine windows
        def out_bound_mask(pt, w, h):
            return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
        ob_mask = out_bound_mask(delta_w_pt0_f_round, W, W)
        nearest_index1[ob_mask] = 0
        correct_0to1_f[ob_mask] = 0

        m_ids, i_ids = torch.where(correct_0to1_f != 0)
        j_ids = nearest_index1[m_ids, i_ids]  # i_ids, j_ids range from [0, WW-1]
        # j_ids_di, j_ids_dj = j_ids // W, j_ids % W  # further get the (i, j) index in fine windows of image1 (right image); j_ids_di, j_ids_dj range from [0, W-1]
        m_ids, i_ids = m_ids.to(torch.long), i_ids.to(torch.long)

        # expec_f_gt will be used as the gt of subpixel-level refinement
        # expec_f_gt = delta_w_pt0_f - delta_w_pt0_f_round
        
        if m_ids.numel() == 0:  # special case: there is no pixel-level gt
            loguru_logger.warning(f"No groundtruth fine match found for local regress: {data['pair_names']}")
            # this won't affect fine-level loss calculation
            data.update({'expec_f': torch.zeros(1, 2, device=device)})
            data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
        # else:
        #     expec_f_gt = expec_f_gt[m_ids, i_ids]
        #     data.update({"expec_f_gt": expec_f_gt})
        #     data.update({"m_ids_f": m_ids,
        #                     "i_ids_f": i_ids,
        #                     "j_ids_f_di": j_ids_di,
        #                     "j_ids_f_dj": j_ids_dj
        #                     })

        # 5. construct a pixel-level gt conf_matrix
        conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device, dtype=torch.bool)
        conf_matrix_f_gt[m_ids, i_ids, j_ids] = 1
        data.update({'conf_matrix_f_gt': conf_matrix_f_gt})
        if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
            # calculate the overlap area between warped pixel and grid pixel as the loss weight.
            w_pt0_f_error = w_pt0_f_error[m_ids, i_ids]
            data.update({'conf_matrix_f_error_gt': w_pt0_f_error})
            
        if  conf_matrix_f_gt.sum() == 0:
            loguru_logger.info(f'no fine matches to supervise')
                
def compute_supervision_fine(data, config, logger=None):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config, logger)
    else:
        raise NotImplementedError
    
@torch.no_grad()
def get_warp_index(
    bs_kpts0, bs_kpts1, E_0to1, E_1to0, K0, K1, bs, s0, s1, w0, w1, obj_geod_th=1e-5
):

    bs_dist = symmetric_epipolar_distance(
        bs_kpts0, bs_kpts1, E_0to1[bs], K0[bs], K1[bs]
    )
    bs_dist_mask = bs_dist <= obj_geod_th
    del bs_dist
    bs_kpts0, bs_kpts1 = bs_kpts0[bs_dist_mask], bs_kpts1[bs_dist_mask]
    # b_ids0, i_ids0, j_ids0
    bs_grid_pt0 = bs_kpts0
    bs_w_pt0 = bs_kpts1
    bs_grid_pt0_c = (bs_kpts0 / s0[bs]).round().long()
    bs_w_pt0_c_long = (bs_kpts1 / s1[bs]).round().long()
    f_bs_i_ids0 = bs_grid_pt0_c[:, 0] + bs_grid_pt0_c[:, 1] * w0
    f_bs_j_ids0 = bs_w_pt0_c_long[:, 0] + bs_w_pt0_c_long[:, 1] * w1
    f_bs_b_ids0 = torch.full_like(f_bs_j_ids0, bs)

    # b_ids1, j_ids1, i_ids1
    f_bs_j_ids1 = (bs_w_pt0_c_long[:, 0] + bs_w_pt0_c_long[:, 1] * w1).unique()
    bs_grid_pt1_c = torch.stack([f_bs_j_ids1 % w1, f_bs_j_ids1 // w1], dim=1)
    bs_grid_pt1 = bs_grid_pt1_c * s1[bs]
    n1 = len(bs_grid_pt1)
    n0 = len(bs_kpts0)
    tomatch_pt1 = rearrange(
        bs_grid_pt1.unsqueeze(1).repeat(1, n0, 1), "n1 n0 c -> (n1 n0) c"
    )
    tomatch_pt0 = rearrange(bs_kpts0.repeat(n1, 1, 1), "n1 n0 c -> (n1 n0) c")
    match_scores = symmetric_epipolar_distance(
        tomatch_pt1, tomatch_pt0, E_1to0[bs], K1[bs], K0[bs]
    )
    v, ind = rearrange(match_scores, "(n1 n0) -> n1 n0", n1=n1, n0=n0).min(dim=1)
    del match_scores, tomatch_pt1, tomatch_pt0
    bs_w_pt1 = bs_kpts0[ind]
    bs_w_pt1_c_long = (bs_w_pt1 / s0[bs]).round().long()
    f_bs_i_ids1 = bs_w_pt1_c_long[:, 0] + bs_w_pt1_c_long[:, 1] * w0
    f_bs_b_ids1 = torch.full_like(f_bs_i_ids1, bs)

    return (
        bs_grid_pt0,
        bs_w_pt0,
        f_bs_b_ids0,
        f_bs_i_ids0,
        f_bs_j_ids0,
        bs_grid_pt1,
        bs_w_pt1,
        f_bs_b_ids1,
        f_bs_j_ids1,
        f_bs_i_ids1,
    )

@torch.no_grad()
def get_scale_gt_matrix5(
    data, scale_l, N, H0, W0, H1, W1, device, require_depth_mask=True, obj_geod_th=1e-5
):
    # pdb.set_trace()
    device = data["K0"].device
    bs = data["K0"].shape[0]
    scale0 = (
        scale_l * data["scale0"][:, None]
        if "scale0" in data
        else torch.tensor(
            [[[scale_l, scale_l]]], dtype=torch.float, device=device
        ).repeat(bs, 1, 1)
    )  # float(scale_l)
    scale1 = (
        scale_l * data["scale1"][:, None]
        if "scale0" in data
        else torch.tensor(
            [[[scale_l, scale_l]]], dtype=torch.float, device=device
        ).repeat(bs, 1, 1)
    )  # float(scale_l)
    h0, w0, h1, w1 = map(lambda x: x // scale_l, [H0, W0, H1, W1])
    scale_wh0_l, scale_wh1_l = (data["scale_wh0"] // scale_l), (
        data["scale_wh1"] // scale_l
    )

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = (
        create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)
    )  # [N, hw, 2] - <x, y>
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = (
        create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    )
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if "mask0_d{}".format(int(scale_l)) in data:
        grid_pt0_i = mask_pts_at_padded_regions(
            grid_pt0_i, data["mask0_d{}".format(int(scale_l))]
        )  # [N, L=h0*w0, 2] - <x, y>
        grid_pt1_i = mask_pts_at_padded_regions(
            grid_pt1_i, data["mask1_d{}".format(int(scale_l))]
        )  # [N, S=h1*w1, 2] - <x, y>

    valid_mask0, w_pt0_i, d_mask0, consistent_mask0 = warp_kpts_ada(
        grid_pt0_i,
        data["depth0"],
        data["depth1"],
        data["T_0to1"],
        data["T_1to0"],
        data["K0"],
        data["K1"],
    )  # 原图尺寸
    valid_mask1, w_pt1_i, d_mask1, consistent_mask1 = warp_kpts_ada(
        grid_pt1_i,
        data["depth1"],
        data["depth0"],
        data["T_1to0"],
        data["T_0to1"],
        data["K1"],
        data["K0"],
    )
   
    s_wh1 = scale_wh1_l.unsqueeze(1).repeat(1, valid_mask0.shape[1], 1)
    w_pt0_all_long = (w_pt0_i / scale1).round().long()
    covisible_mask0 = (
        (w_pt0_all_long[..., 0] > 0)
        * (w_pt0_all_long[..., 0] < s_wh1[..., 0] - 1)
        * (w_pt0_all_long[..., 1] > 0)
        * (w_pt0_all_long[..., 1] < s_wh1[..., 1] - 1)
    )
   
    s_wh0 = scale_wh0_l.unsqueeze(1).repeat(1, valid_mask1.shape[1], 1)
    w_pt1_all_long = (w_pt1_i / scale0).round().long()
    covisible_mask1 = (
        (w_pt1_all_long[..., 0] > 0)
        * (w_pt1_all_long[..., 0] < s_wh0[..., 0] - 1)
        * (w_pt1_all_long[..., 1] > 0)
        * (w_pt1_all_long[..., 1] < s_wh0[..., 1] - 1)
    )
    flag = 0
    if (valid_mask0 * covisible_mask0 * consistent_mask0 == 0).all() and (
        valid_mask1 * covisible_mask1 * consistent_mask1 == 0
    ).all():
        flag = 1
        valid_mask0 = valid_mask0 * covisible_mask0
        valid_mask1 = valid_mask1 * covisible_mask1
    else:
        valid_mask0 = valid_mask0 * consistent_mask0 * covisible_mask0
        valid_mask1 = valid_mask1 * consistent_mask1 * covisible_mask1
    del covisible_mask0, covisible_mask1, consistent_mask0, consistent_mask1

    b_ids0, i_ids0 = torch.nonzero(valid_mask0, as_tuple=True)
    v_w_pt0_i_long = w_pt0_all_long[
        b_ids0, i_ids0
    ]  # (w_pt0_i / scale1)[b_ids0, i_ids0].round().long()
    j_ids0 = v_w_pt0_i_long[:, 0] + v_w_pt0_i_long[:, 1] * w1
   

    b_ids1, j_ids1 = torch.nonzero(valid_mask1, as_tuple=True)
    v_w_pt1_i_long = w_pt1_all_long[
        b_ids1, j_ids1
    ]  # (w_pt1_i / scale0)[b_ids1, j_ids1].round().long()
    i_ids1 = v_w_pt1_i_long[:, 0] + v_w_pt1_i_long[:, 1] * w0
   

    f_b_ids0, f_i_ids0, f_j_ids0, f_b_ids1, f_j_ids1, f_i_ids1, = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    f_grid_pt0, f_w_pt0, f_grid_pt1, f_w_pt1 = [], [], [], []
    E_0to1, F_0to1 = pose2essential_fundamental(data["K0"], data["K1"], data["T_0to1"])
    E_1to0, F_1to0 = pose2essential_fundamental(data["K1"], data["K0"], data["T_1to0"])

    for bs in range(N):
        bs_mask0 = b_ids0 == bs
        bs_b_ids0, bs_i_ids0, bs_j_ids0 = (
            b_ids0[bs_mask0],
            i_ids0[bs_mask0],
            j_ids0[bs_mask0],
        )
        bs_mask1 = b_ids1 == bs
        bs_b_ids1, bs_j_ids1, bs_i_ids1 = (
            b_ids1[bs_mask1],
            j_ids1[bs_mask1],
            i_ids1[bs_mask1],
        )

        if len(bs_i_ids0) == 0 or len(bs_j_ids1) == 0:
            try:
                if len(bs_i_ids0) > len(bs_j_ids1):
                    bs_kpts0 = grid_pt0_i[bs_b_ids0, bs_i_ids0]
                    bs_kpts1 = w_pt0_i[bs_b_ids0, bs_i_ids0]
                    (
                        bs_grid_pt0,
                        bs_w_pt0,
                        f_bs_b_ids0,
                        f_bs_i_ids0,
                        f_bs_j_ids0,
                        bs_grid_pt1,
                        bs_w_pt1,
                        f_bs_b_ids1,
                        f_bs_j_ids1,
                        f_bs_i_ids1,
                    ) = get_warp_index(
                        bs_kpts0,
                        bs_kpts1,
                        E_0to1,
                        E_1to0,
                        data["K0"],
                        data["K1"],
                        bs,
                        scale0,
                        scale1,
                        w0,
                        w1,
                        obj_geod_th,
                    )
                else:
                    bs_kpts1 = grid_pt1_i[bs_b_ids1, bs_j_ids1]
                    bs_kpts0 = w_pt1_i[bs_b_ids1, bs_j_ids1]
                    (
                        bs_grid_pt1,
                        bs_w_pt1,
                        f_bs_b_ids1,
                        f_bs_j_ids1,
                        f_bs_i_ids1,
                        bs_grid_pt0,
                        bs_w_pt0,
                        f_bs_b_ids0,
                        f_bs_i_ids0,
                        f_bs_j_ids0,
                    ) = get_warp_index(
                        bs_kpts1,
                        bs_kpts0,
                        E_1to0,
                        E_0to1,
                        data["K1"],
                        data["K0"],
                        bs,
                        scale1,
                        scale0,
                        w1,
                        w0,
                        obj_geod_th,
                    )
            except:
                print(data["scene_id"], data["pair_id"], data["pair_names"])
                bs_grid_pt0 = grid_pt0_i[bs_b_ids0, bs_i_ids0]
                bs_w_pt0 = w_pt0_i[bs_b_ids0, bs_i_ids0]
                f_bs_b_ids0, f_bs_i_ids0, f_bs_j_ids0 = bs_b_ids0, bs_i_ids0, bs_j_ids0
                bs_grid_pt1 = grid_pt1_i[bs_b_ids1, bs_j_ids1]
                bs_w_pt1 = w_pt1_i[bs_b_ids1, bs_j_ids1]
                f_bs_b_ids1, f_bs_j_ids1, f_bs_i_ids1 = bs_b_ids1, bs_j_ids1, bs_i_ids1
        else:
            bs_grid_pt0 = grid_pt0_i[bs_b_ids0, bs_i_ids0]
            bs_w_pt0 = w_pt0_i[bs_b_ids0, bs_i_ids0]
            f_bs_b_ids0, f_bs_i_ids0, f_bs_j_ids0 = bs_b_ids0, bs_i_ids0, bs_j_ids0

            bs_grid_pt1 = grid_pt1_i[bs_b_ids1, bs_j_ids1]
            bs_w_pt1 = w_pt1_i[bs_b_ids1, bs_j_ids1]
            f_bs_b_ids1, f_bs_j_ids1, f_bs_i_ids1 = bs_b_ids1, bs_j_ids1, bs_i_ids1

        geod_mask0 = (
            symmetric_epipolar_distance(
                bs_grid_pt0, bs_w_pt0, E_0to1[bs], data["K0"][bs], data["K1"][bs]
            )
            <= obj_geod_th
        )
        geod_mask1 = (
            symmetric_epipolar_distance(
                bs_w_pt1, bs_grid_pt1, E_0to1[bs], data["K0"][bs], data["K1"][bs]
            )
            <= obj_geod_th
        )

        f_b_ids0.append(f_bs_b_ids0[geod_mask0])
        f_i_ids0.append(f_bs_i_ids0[geod_mask0])
        f_j_ids0.append(f_bs_j_ids0[geod_mask0])
        f_grid_pt0.append(bs_grid_pt0[geod_mask0])
        f_w_pt0.append(bs_w_pt0[geod_mask0])

        f_b_ids1.append(f_bs_b_ids1[geod_mask1])
        f_j_ids1.append(f_bs_j_ids1[geod_mask1])
        f_i_ids1.append(f_bs_i_ids1[geod_mask1])
        f_grid_pt1.append(bs_grid_pt1[geod_mask1])
        f_w_pt1.append(bs_w_pt1[geod_mask1])

    f_b_ids0 = torch.cat(f_b_ids0, dim=0)
    f_i_ids0 = torch.cat(f_i_ids0, dim=0)
    f_j_ids0 = torch.cat(f_j_ids0, dim=0)
    f_grid_pt0_i = torch.cat(f_grid_pt0, dim=0)
    f_w_pt0_i = torch.cat(f_w_pt0, dim=0)

    f_b_ids1 = torch.cat(f_b_ids1, dim=0)
    f_j_ids1 = torch.cat(f_j_ids1, dim=0)
    f_i_ids1 = torch.cat(f_i_ids1, dim=0)
    f_grid_pt1_i = torch.cat(f_grid_pt1, dim=0)
    f_w_pt1_i = torch.cat(f_w_pt1, dim=0)

    w_pt0_i[f_b_ids0, f_i_ids0] = f_w_pt0_i
    w_pt1_i[f_b_ids1, f_j_ids1] = f_w_pt1_i

    if require_depth_mask:
        # pdb.set_trace()
        depth_mask0 = torch.zeros_like(d_mask0).bool()
        depth_mask0[f_b_ids0, f_i_ids0] = True
        depth_mask1 = torch.zeros_like(d_mask1).bool()
        depth_mask1[f_b_ids1, f_j_ids1] = True

        return (
            f_b_ids0,
            f_i_ids0,
            f_j_ids0,
            f_b_ids1,
            f_j_ids1,
            f_i_ids1,
            depth_mask0,
            depth_mask1,
            w_pt0_i,
            grid_pt0_i,
            w_pt1_i,
            grid_pt1_i,
        )
      
    else:
        return (
            f_b_ids0,
            f_i_ids0,
            f_j_ids0,
            f_b_ids1,
            f_j_ids1,
            f_i_ids1,
            w_pt0_i,
            grid_pt0_i,
            w_pt1_i,
            grid_pt1_i,
        )