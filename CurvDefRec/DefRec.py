import torch
import numpy as np
import torch.nn as nn
import utils.pc_utils as pc_utils

DefRec_SCALER = 20.0

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


class Group_Region(nn.Module):  # Region-based Grouping

    def __init__(self, nregions=3, largest=True):
        super().__init__()
        self.num_group = nregions ** 3
        self.nregions = nregions
        self.largest = largest

    def forward(self, xyz, curv):
        '''
            input: B N 3, B N 1
            ---------------------------
            output: B G, B N
        '''
        batch_size, _, _ = xyz.size()
        
        # Assign points to regions
        region_labels = pc_utils.assign_region_to_point(xyz.permute(0, 2, 1), xyz.device, self.nregions)  # B N

        region_curv_diversity = [[] for _ in range(batch_size)]
        valid_regions = [[] for _ in range(batch_size)]

        for batch in range(batch_size):
            for region in range(self.num_group):
                region_mask = (region_labels[batch] == region)  # R
                region_curv = curv[batch, region_mask]  # R

                if region_curv.size(0) == 0:
                    continue
                
                curv_norm = (region_curv - region_curv.min()) / (region_curv.max() - region_curv.min() + 1e-10)
                curv_entropy = -torch.sum(curv_norm * torch.log(curv_norm + 1e-10)).unsqueeze(0)

                region_curv_diversity[batch].append(region_curv.std().unsqueeze(0) * 100 + curv_entropy)
                valid_regions[batch].append(region)

        return None, region_curv_diversity, region_labels, valid_regions

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
        
        idx = idx.view(batch_size, self.num_group, self.group_size)
        idx = idx - idx_base
        
        # Normalize curvatures
        curv_min = curv.min(dim=2, keepdim=True)[0]
        curv_max = curv.max(dim=2, keepdim=True)[0]
        curv_norm = (curv - curv_min) / (curv_max - curv_min + 1e-10)

        # Compute entropy of normalized curvatures in each group
        curv_entropy = -torch.sum(curv_norm * torch.log(curv_norm + 1e-10), dim=2, keepdim=True)  # B G 1

        return neighborhood, curv_entropy, idx

def deform_input_curv(X, curvs, DefRec_dist='volume_based_voxels',
                      top_k=5, num_group=20, group_size=55, largest=False):
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

    group_layer = Group(num_group=num_group, group_size=group_size).to(X.device)
    _, curv_eval, idx = group_layer(X, curvs)
    B, N, _ = X.size()
    _, G, M = idx.size()
    mask = torch.zeros(B, N, 3).to(X.device)  # binary mask of deformed points

    # Normalize curv_std to use as probabilities
    curv_eval_prob = torch.nn.functional.softmax(curv_eval, dim=1)

    for b in range(B):
        # Get the top 3 regions based on the highest curv_std probabilities
        top_indices = torch.topk(curv_eval_prob[b, :, 0], top_k, largest=largest).indices.cpu().numpy()
        
        selected_indices = top_indices
        
        for g in selected_indices:

            points_idx = idx[b, g]
            
            # Deform this neighborhood
            center = X[b, points_idx].mean(dim=0).cpu().numpy()
            rnd_pts = pc_utils.draw_from_gaussian(center, M)
            X[b, points_idx, :3] = torch.tensor(rnd_pts.T, dtype=torch.float).to(X.device)
            mask[b, points_idx, :] = 1

    return X.permute(0,2,1), mask.permute(0,2,1)


def deform_input(X, lookup, DefRec_dist='volume_based_voxels', device='cuda:0'):
    """
    Deform a region in the point cloud. For more details see https://arxiv.org/pdf/2003.12641.pdf
    Input:
        args - commmand line arguments
        X - Point cloud [B, C, N]
        lookup - regions center point
        device - cuda/cpu
    Return:
        X - Point cloud with a deformed region
        mask - 0/1 label per point indicating if the point was centered
    """

    # get points' regions 
    regions = pc_utils.assign_region_to_point(X, device)

    n = pc_utils.NREGIONS
    min_pts = 40
    region_ids = np.random.permutation(n ** 3)
    mask = torch.zeros_like(X).to(device)  # binary mask of deformed points

    for b in range(X.shape[0]):
        if DefRec_dist == 'volume_based_radius':
            X[b, :, :], indices = pc_utils.collapse_to_point(X[b, :, :], device)
            mask[b, :3, indices] = 1
        else:
            for i in region_ids:
                ind = regions[b, :] == i
                # if there are enough points in the region
                if torch.sum(ind) >= min_pts:
                    region = lookup[i].cpu().numpy()  # current region average point
                    mask[b, :3, ind] = 1
                    num_points = int(torch.sum(ind).cpu().numpy())
                    if DefRec_dist == 'volume_based_voxels':
                        rnd_pts = pc_utils.draw_from_gaussian(region, num_points)
                        X[b, :3, ind] = torch.tensor(rnd_pts, dtype=torch.float).to(device)
                    break  # move to the next shape in the batch
    return X, mask


def chamfer_distance(p1, p2, mask):

    """
    Calculate Chamfer Distance between two point sets
    Input:
        p1: size[B, C, N]
        p2: size[B, C, N]
    Return: 
        sum of all batches of Chamfer Distance of two point sets
    """

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    # add dimension
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    # repeat point values at the new dimension
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    # calc norm between each point in p1 and each point in p2
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3) ** 2

    # add big value to points not in voxel and small 0 to those in voxel
    mask_cord = mask[:, :, 0]  # take only one coordinate  (batch_size, #points)
    m = mask_cord.clone()
    m[m == 0] = 100  # assign big value to points not in the voxel
    m[m == 1] = 0
    m = m.view(dist.size(0), 1, dist.size(2))  # transform to (batch_size, 1, #points)
    dist = dist + m

    # take the minimum distance for each point in p1 and sum over batch
    dist = torch.min(dist, dim=2)[0]  # for each point in p1 find the min in p2 (takes only from relevant ones because of the previous step)
    sum_pc = torch.sum(dist * mask_cord, dim=1)  # sum distances of each example (array broadcasting - zero distance of points not in the voxel for p1 and sum distances)
    dist = torch.sum(torch.div(sum_pc, torch.sum(mask_cord, dim=1)))  # divide each pc with the number of active points and sum
    
    return dist


def reconstruction_loss(pred, gold, mask):

    """
    Calculate symmetric chamfer Distance between predictions and labels
    Input:
        pred: size[B, C, N]
        gold: size[B, C, N]
        mask: size[B, C, N]
    Return: 
        mean batch loss
    """
    gold = gold.clone()

    batch_size = pred.size(0)

    # [batch_size, #points, coordinates]
    gold = gold.permute(0, 2, 1)
    mask = mask.permute(0, 2, 1)

    # calc average chamfer distance for each direction
    dist_gold = chamfer_distance(gold, pred, mask)
    dist_pred = chamfer_distance(pred, gold, mask)
    chamfer_loss = dist_gold + dist_pred

    # average loss
    loss = (1 / batch_size) * chamfer_loss

    return loss


def calc_loss(args, logits, labels, mask):

    """
    Calc. DefRec loss.
    Return: loss 
    """

    prediction = logits['DefRec']
    loss = args.DefRec_weight * reconstruction_loss(prediction, labels, mask) * DefRec_SCALER
    
    return loss


def displacement_loss(pred, target, mask):
    """
    Calculate L2 loss between predicted displacements and ground truth displacements
    Input:
        pred: size[B, 3, N] - predicted displacements
        target: size[B, 3, N] - ground truth displacements
        mask: size[B, 3, N] - mask indicating valid points
    Return: 
        L2 loss between predicted and ground truth displacements
    """

    # Calculate L2 loss only for valid points (mask == 1)
    loss = ((pred - target) ** 2) * mask
    loss = loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
    loss = loss.mean()

    return loss

def calc_loss_dist(args, logits, labels, mask):

    """
    Calc. DefRec loss.
    Return: loss 
    """

    prediction = logits['DefRec']
    loss = args.DefRec_weight * displacement_loss(prediction, labels, mask) * DefRec_SCALER
    
    return loss
