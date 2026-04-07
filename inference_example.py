"""
Example usage of InferenceDataLoader with NSM model for inference.

This demonstrates:
1. Loading vocabularies and model
2. Initializing InferenceDataLoader
3. Running inference on a single query
4. Processing results
"""

import torch
import numpy as np
import os
from NSM.data.inference_loader import InferenceDataLoader
from NSM.data.basic_dataset import load_dict
# from infer_nsm import infer_top_k
import argparse
from parse_args import parser
from NSM.util.utils import create_logger
from hybrid_inference_interface import HybridNSMInference

def load_model_and_vocab(args):
    """Load pre-trained model and vocabularies."""
    
    # Load vocabularies
    data_folder = args.data_folder
    word2id = load_dict(os.path.join(data_folder, args.word2id))
    relation2id = load_dict(os.path.join(data_folder, args.relation2id))
    entity2id = load_dict(os.path.join(data_folder, args.entity2id))
    
    print(f"Loaded vocabularies:")
    print(f"  Words: {len(word2id)}")
    print(f"  Relations: {len(relation2id)}")
    print(f"  Entities: {len(entity2id)}")
    
    return word2id, relation2id, entity2id


def format_results(results, id2entity, num_to_show=5):
    """Pretty print inference results."""
    print("\n" + "="*80)
    print(f"Query: {results['query']}")
    print("="*80)
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    top_results = results['results']
    
    print(f"Top {num_to_show} candidates:")
    print(f"{'Rank':<6} {'Entity':<40} {'Probability':<12}")
    print("-"*60)
    
    for rank, (entity_id, prob) in enumerate(top_results['top_k_candidates'][:num_to_show], 1):
        entity_name = id2entity.get(entity_id, f"UNKNOWN_{entity_id}")
        print(f"{rank:<6} {entity_name:<40} {prob:.6f}")
    
    print()


def demo_inference():
    """Demonstrate the inference workflow."""
    
    args = parser.parse_args()
    
    # ========================================================================
    # Step 1: Load vocabularies
    # ========================================================================
    print("\n[Step 1] Loading vocabularies...")
    word2id, relation2id, entity2id = load_model_and_vocab(args)
    # id2entity = {idx: ent for ent, idx in entity2id.items()}
    
    # ========================================================================
    # Step 2: Initialize InferenceDataLoader
    # ========================================================================
    print("\n[Step 2] Initializing InferenceDataLoader...")

    
    inference_loader = InferenceDataLoader(args, word2id, relation2id, entity2id)
    print(f"Available subgraphs: {len(inference_loader.list_available_subgraphs())}")
    
    # ========================================================================
    # Step 3: Load model (dummy example - in practice load your actual model)
    # ========================================================================
    print("\n[Step 3] Loading model...")
    args.use_cuda = torch.cuda.is_available()
    logger = create_logger(args)
    
    hybrid_inferencer = HybridNSMInference(
        args=args,
        logger=logger,
        loader=inference_loader,
        model_checkpoint=os.path.join(args.checkpoint_dir, args.load_experiment),
        device='cuda'
    )

    # print("Model loading skipped (use your actual model)")
    print("\nInferenceDataLoader is ready for use!")
    
    # ========================================================================
    # Step 4: Example queries (you would replace these with real data)
    # ========================================================================
    print("\n[Step 4] Example inference queries:")
    print("-" * 80)
    
    # Example 1: Direct batch tuple usage
    if inference_loader.list_available_subgraphs():
        subgraph_id = inference_loader.list_available_subgraphs()[0]
        print(f"\nUsing subgraph: {subgraph_id}")
        
        res = hybrid_inferencer.infer(
            subgraph_id=subgraph_id,
            query="Who is the president of France?",
            seed_entity_id=1079,  # Replace with actual entity ID
            top_k=10,
        )
        print("Answers returned:", res)

if __name__ == '__main__':
    demo_inference()
