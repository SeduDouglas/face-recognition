import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, squared=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared
        
    def forward(self, embeddings, labels):
        dist_matrix = self._distance_matrix(embeddings)

        # Get anchor positive and anchor negative masks
        ap_mask = self._get_anchor_positive_mask(labels)
        an_mask = self._get_anchor_negative_mask(labels)

        # Get positive and negative distances
        ap_distances = dist_matrix * ap_mask
        an_distances = dist_matrix * an_mask
        an_distances[torch.logical_not(an_mask)] = float('inf')
        # For each anchor, get the hardest positive
        hardest_positive_dist = ap_distances.max(dim=1)[0]
        index_invalid_positive = hardest_positive_dist == 0
        hardest_positive_dist[index_invalid_positive] = -10
        
        # For each anchor, get the semi-hard negatives
        # Semi-hard negatives: negatives that are farther than the positive exemplar
        # but still within the margin
        mask_semi_hard_negatives = an_mask * torch.logical_and(
            dist_matrix > hardest_positive_dist.unsqueeze(1),
            dist_matrix < hardest_positive_dist.unsqueeze(1) + self.margin
        )
        
        # If no semi-hard negatives exist, use the hardest negatives
        if mask_semi_hard_negatives.sum() == 0:
            hardest_negative_dist = an_distances.min(dim=1)[0]
        else:
            # Get the semi-hard negative distances
            semi_hard_negatives = dist_matrix * mask_semi_hard_negatives
            semi_hard_negatives[semi_hard_negatives == 0] = float('inf')
            hardest_negative_dist = semi_hard_negatives.min(dim=1)[0]

        # Calculate triplet loss
        triplet_loss = torch.clamp(
            hardest_positive_dist - hardest_negative_dist + self.margin,
            min=0.0
        )

        triplet_loss = triplet_loss[torch.logical_not(index_invalid_positive)]
        num_valid_triplets = len(triplet_loss)
        
        if num_valid_triplets == 0:
            return torch.tensor(0.0, device=embeddings.device), num_valid_triplets

        triplet_loss = triplet_loss.mean()
        return triplet_loss, num_valid_triplets
    
    def _distance_matrix(self, embeddings):
        pdist = torch.cdist(embeddings, embeddings, p=2)
        
        return pdist
    
    def _get_anchor_positive_mask(self, labels):
        return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    def _get_anchor_negative_mask(self, labels):
        return (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
