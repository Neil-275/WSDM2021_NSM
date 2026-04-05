import torch
import numpy as np


def infer_top_k(student, batch, id2entity, eps=0.05, top_k=10):
    """
    Inference function for NSM: returns top-k candidate entities with highest scores.
    
    Args:
        student: Trained NSM model (should be in eval mode)
        batch: Batch data from dataloader, format: 
               (local_entity, query_entities, kb_adj_mat, query_text, 
                seed_dist, true_batch_id) - note: without answer_list
        id2entity: Dictionary mapping entity index to entity name
        eps: Probability threshold (used to filter low-probability candidates)
        top_k: Number of top candidates to return
    
    Returns:
        List of dicts, one per sample in batch:
        [
            {
                'top_k_candidates': [(entity_id, score), ...],  # sorted by score descending
                'entity_names': [entity_name, ...],  # human-readable entity names
                'all_candidates': [(entity_id, score), ...],  # all candidates above eps threshold
            },
            ...
        ]
    """
    student.eval()
    results = []
    
    with torch.no_grad():
        # Forward pass through model
        loss, extras, pred_dist, tp_list = student(batch)
    
    # Extract batch data
    local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id = batch
    
    batch_size = pred_dist.size(0)
    max_local_entity = local_entity.shape[1]
    pad_ent_id = len(id2entity)
    
    # Probability threshold for filtering
    ignore_prob = (1 - eps) / max_local_entity
    
    for batch_id in range(batch_size):
        # Get predictions and metadata for this sample
        candidates = local_entity[batch_id, :].tolist()
        probs = pred_dist[batch_id, :].tolist()
        seed_entities = query_entities[batch_id, :].tolist()
        
        # Filter candidates
        candidate2prob = []
        for c, p, s in zip(candidates, probs, seed_entities):
            # Skip seed entities (query entities)
            if s == 1.0:
                continue
            # Skip padding entity index
            if c == pad_ent_id:
                continue
            # Skip very low probability candidates
            if p < ignore_prob:
                continue
            candidate2prob.append((c, p))
        
        # Sort by probability (descending)
        candidate2prob_sorted = sorted(candidate2prob, key=lambda x: x[1], reverse=True)
        
        # Get top-k
        top_k_candidates = candidate2prob_sorted[:top_k]
        
        # Convert entity IDs to entity names
        entity_names = []
        for ent_id, score in top_k_candidates:
            entity_names.append(id2entity.get(ent_id, f"UNKNOWN_{ent_id}"))
        
        results.append({
            'top_k_candidates': top_k_candidates,
            'entity_names': entity_names,
            'all_candidates': candidate2prob_sorted,  # All candidates that passed filtering
        })
    
    return results


def infer_single_sample(student, sample_data, id2entity, max_local_entity,
                        eps=0.05, top_k=10, device='cuda'):
    """
    Convenience wrapper for single sample inference.
    
    Args:
        student: Trained NSM model
        sample_data: Single sample batch from dataloader, format:
                     (local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id)
        id2entity: Entity index to name mapping
        max_local_entity: Max number of local entities in the graph
        eps: Probability threshold
        top_k: Number of top candidates to return
        device: Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        dict with 'top_k_candidates', 'entity_names', 'all_candidates'
    """
    student.to(device)
    student.eval()
    
    # Move batch data to device
    batch_on_device = tuple(
        torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x 
        for x in sample_data
    )
    
    results = infer_top_k(student, batch_on_device, id2entity, eps=eps, top_k=top_k)
    
    return results[0] if results else None
