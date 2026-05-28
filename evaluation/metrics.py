import torch
import numpy as np

def cosine_similarity(f1, f2):
    f1 = f1 / f1.norm(dim=1, keepdim=True)
    f2 = f2 / f2.norm(dim=1, keepdim=True)
    return torch.mm(f1, f2.t())

def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    
    for q_idx in range(num_q):
        valid_g = g_camids != q_camids[q_idx]
        if not np.any(valid_g): continue
        orig_cmc = matches[q_idx][valid_g[indices[q_idx]]]
        if not np.any(orig_cmc): continue
        
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    assert num_valid_q > 0
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP
