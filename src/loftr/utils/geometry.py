import torch
from kornia.geometry.epipolar import  numeric

@torch.no_grad()
def skew(v):
    # The skew-symmetric matrix of vector
    return torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=v.device
    )


@torch.no_grad()
def pose2fundamental(K0, K1, T_0to1):
    # pdb.set_trace()
    Tx = numeric.cross_product_matrix(T_0to1[:, :3, 3])
    E_mat = Tx @ T_0to1[:, :3, :3]
    # F = torch.inverse(K1).T @ R0to1 @ K0.T @ skew((K0 @ R0to1.T).dot(t0to1.reshape(3,)))
    # F_mat = torch.inverse(K1).T @ E_mat @ torch.inverse(K0)
    # F_mat = fundamental.fundamental_from_essential(E_mat, K0, K1)
    F_mat = torch.inverse(K1).transpose(1, 2) @ E_mat @ torch.inverse(K0)
    return F_mat


@torch.no_grad()
def pose2essential_fundamental(K0, K1, T_0to1):
    # pdb.set_trace()
    Tx = numeric.cross_product_matrix(T_0to1[:, :3, 3])
    E_mat = Tx @ T_0to1[:, :3, :3]
    F_mat = torch.inverse(K1).transpose(1, 2) @ E_mat @ torch.inverse(K0)
    return E_mat, F_mat


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0

@torch.no_grad()
def warp_kpts_ada(kpts0, depth0, depth1, T_0to1=None, T_1to0=None, K0=None, K1=None):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W], depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3], K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
        depth_mask
    """
    # kpts0_depth = interpolate_depth(kpts0, depth0)
    # pdb.set_trace()
    kpts0_long = kpts0.round().long()
    kpts0_depth = torch.stack(
        [
            depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]]
            for i in range(kpts0.shape[0])
        ],
        dim=0,
    )  # (N, L)
    depth_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    # w_kpts0_long = w_kpts0.long()
    # w_kpts0_long[~covisible_mask, :] = 0
    # w_kpts0_depth = interpolate_depth(w_kpts0, depth1)
    w_kpts0_long = w_kpts0.round().long()
    w_kpts0_long[~covisible_mask, :] = 0
    w_kpts0_depth = torch.stack(
        [
            depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]]
            for i in range(w_kpts0.shape[0])
        ],
        dim=0,
    )
    if T_1to0 is None:
        consistent_mask = (
            (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
        ).abs() < 0.2  # 0.2
        valid_mask = depth_mask * covisible_mask * consistent_mask

        return valid_mask, w_kpts0, depth_mask
    else:
        kpts1_h = (
            torch.cat([w_kpts0, torch.ones_like(w_kpts0[:, :, [0]])], dim=-1)
            * w_kpts0_depth[..., None]
        )
        kpts1_cam = K1.inverse() @ kpts1_h.transpose(2, 1)  # (N, 3, L)
        w_kpts1_cam = T_1to0[:, :3, :3] @ kpts1_cam + T_1to0[:, :3, [3]]
        w_kpts1_h = (K0 @ w_kpts1_cam).transpose(2, 1)  # (N, L, 3)
        w_kpts1 = w_kpts1_h[:, :, :2] / (w_kpts1_h[:, :, [2]] + 1e-4)  # (N, L, 2)
        consistent_mask = torch.norm(w_kpts1 - kpts0, p=2, dim=-1) < 4.0  # 4.  5.

        # pdb.set_trace()
        # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.5 # 0.2  # 0.2
        valid_mask = depth_mask * covisible_mask  # * consistent_mask

        return valid_mask, w_kpts0, depth_mask, consistent_mask