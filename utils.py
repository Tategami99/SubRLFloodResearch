import torch
import numpy as np

def create_action_mask(visited_locations, possible_locations):
    mask = torch.ones(len(possible_locations), dtype=torch.float32)
    for idx, loc in enumerate(possible_locations):
        if loc in visited_locations:
            mask[idx] = 0.0 # invalid (already occupied)
    return mask

def apply_action_mask(action_probs, action_mask, epsilon=1e-8):
    """Apply action mask and renormalize probabilities"""
    masked_probs = action_probs * action_mask
    
    # If all actions are masked, use uniform distribution over valid actions
    if masked_probs.sum() <= epsilon:
        valid_count = action_mask.sum()
        if valid_count > 0:
            masked_probs = action_mask / valid_count
        else:
            # Fallback: uniform over all actions (shouldn't happen normally)
            masked_probs = torch.ones_like(action_probs) / len(action_probs)
    else:
        masked_probs = masked_probs / masked_probs.sum()
    
    return masked_probs
