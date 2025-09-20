import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.pc_utils as pc_utils

def fps(xyz, npoint):

    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)
    centroids_vals = torch.zeros(B, C, npoint).to(xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    
    return centroids_vals.permute(0,2,1).contiguous()

def knn(xyz, centers, k):
    xx = torch.sum(xyz ** 2, dim=2, keepdim=True)  # B x N x 1
    center_sq = torch.sum(centers ** 2, dim=2).unsqueeze(1)  # B x 1 x M
    inner = -2 * torch.matmul(xyz, centers.transpose(2, 1))  # B x N x M
    pairwise_distance = xx + center_sq + inner  # B x N x M
    _, idx = torch.topk(pairwise_distance, k, dim=1, largest=False, sorted=True)  # B x k x M   
    return idx

class Group(nn.Module):  # FPS + KNN

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, curv):
        '''
            input: B N 3, B N 1
            ---------------------------
            output: B G M 3, B G M 1, B G 1
            center : B G 3, B G 1
        '''
        batch_size, num_points, _ = xyz.size()
        # fps the centers out
        centers = fps(xyz.permute(0,2,1), self.num_group) # B G 3
        # knn to get the neighborhood
        idx = knn(xyz, centers, self.group_size) # B G M
        idx = idx.transpose(2, 1)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.reshape(-1)
        
        # Process xyz
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        
        # Process curv
        curv = curv.view(batch_size * num_points, -1)[idx, :]
        curv = curv.view(batch_size, self.num_group, self.group_size).contiguous()
        
        # Compute standard deviation of curvatures in each group
        curv_std = curv.std(dim=2, keepdim=True) # B G 1
        idx = idx.view(batch_size, self.num_group, self.group_size)
        idx = idx - idx_base
        
        # # Normalize curvatures
        # curv_min = curv.min(dim=2, keepdim=True)[0]
        # curv_max = curv.max(dim=2, keepdim=True)[0]
        # curv_norm = (curv - curv_min) / (curv_max - curv_min + 1e-10)

        # # Compute entropy of normalized curvatures in each group
        # curv_entropy = -torch.sum(curv_norm * torch.log(curv_norm + 1e-10), dim=2, keepdim=True)  # B G 1

        return neighborhood, curv_std*100, idx

def density(pc, v_point=np.array([1, 0, 0]), gate=1):
    dist = np.sqrt((v_point ** 2).sum())
    max_dist = dist + 1
    min_dist = dist - 1
    dist = np.linalg.norm(pc - v_point.reshape(1,3), axis=1)
    dist = (dist - min_dist) / (max_dist - min_dist)
    r_list = np.random.uniform(0, 1, pc.shape[0])
    tmp_pc = pc[dist * gate < (r_list)]
    return tmp_pc

def drop_hole(pc, p):
    random_point = np.random.randint(0, pc.shape[0])
    index = np.linalg.norm(pc - pc[random_point].reshape(1,3), axis=1).argsort()
    return pc[index[int(pc.shape[0] * p):]]

def aug_pc_curv(X, curvs, aug_method='dropping',
                      top_k=4, remain_k=6):
    """
    Deform a region in the point cloud based on curvature standard deviation.
    Input:
        X - Original point cloud [B, N, 3]
        idx - Indices of the neighborhoods [B, G, M]
        curv_std - Standard deviation of curvature in each group [B, G, 1]
        device - cuda/cpu
    Return:
        X - Point cloud with a deformed region
        mask - 0/1 label per point indicating if the point was centered
    """

    group_layer = Group(num_group=20, group_size=55).to(X.device)
    _, curv_eval, idx = group_layer(X, curvs)

    B, N, _ = X.size()
    _, G, _ = idx.size()
    mask = torch.zeros(B, N, 3).to(X.device)  # binary mask of deformed points

    # Normalize curv_std to use as probabilities
    curv_eval_prob = torch.nn.functional.softmax(curv_eval, dim=1)

    for b in range(B):
        # Get the top 3 regions based on the highest curv_std probabilities
        top_indices = torch.topk(curv_eval_prob[b, :, 0], top_k).indices.cpu().numpy()

        # Get remaining indices after selecting top 3
        remaining_indices = np.setdiff1d(np.arange(G), top_indices)

        # Randomly select 5 regions from the remaining indices
        selected_indices = np.random.choice(remaining_indices, size=remain_k, replace=False)

        for g in selected_indices:
            points_idx = idx[b, g]
            # Deform this neighborhood
            original_points = X[b, points_idx, :].cpu().numpy()
            if aug_method == "dropping":
                X[b, points_idx, :], _ = pc_utils.collapse_to_point(X[b, points_idx, :], X.device)
                mask[b, points_idx, :] = 1
            elif aug_method == "density":
                deformed_points = density(original_points)
                X[b, points_idx, :] = 0
                X[b, points_idx[:deformed_points.shape[0]], :] = torch.tensor(deformed_points).to(X.device)
                X[b, points_idx[deformed_points.shape[0]:], :], _ = pc_utils.collapse_to_point(X[b, points_idx[deformed_points.shape[0]:], :], X.device)
                mask[b, points_idx[deformed_points.shape[0]:], :] = 1

    return X.permute(0,2,1), mask.permute(0,2,1)


# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, z_i, z_j):
#         z_i = z_i["emb"]
#         z_j = z_j["emb"]
#         z_i = F.normalize(z_i, dim=1)
#         z_j = F.normalize(z_j, dim=1)

#         # Cosine similarity matrix
#         sim_matrix = torch.mm(z_i, z_j.T) / self.temperature
#         sim_matrix = torch.exp(sim_matrix)
        
#         # Positive pairs
#         positive_pairs = torch.diag(sim_matrix)

#         # Loss calculation
#         loss = -torch.log(positive_pairs / sim_matrix.sum(dim=1)).mean()
#         return loss
