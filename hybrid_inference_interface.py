import argparse

import torch
import torch.nn as nn
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import re
from NSM.util.utils import create_logger
from parse_args import parser
from NSM.data.inference_loader import InferenceDataLoader

class HybridNSMInference:
    """
    Interface for NSM hybrid model inference on stored subgraphs.
    
    This class assumes that:
    - All subgraphs are pre-computed and stored with their IDs
    - Metadata files (entities.txt, relations.txt, vocab.txt, etc.) are available
    - A trained model checkpoint is provided
    - During inference, the model receives: subgraph_id, query, and returns top-k answers
    
    Attributes:
        model: The trained NSM student model
        entity2id: Dictionary mapping entity names to their indices
        id2entity: Dictionary mapping entity indices to their names
        relation2id: Dictionary mapping relation names to their indices
        word2id: Dictionary mapping words to their indices
        id2word: Dictionary mapping word indices to their words
        subgraph_store: Dictionary storing pre-computed subgraphs indexed by ID
        device: Device for model inference (cuda or cpu)
    """
    
    def __init__(
        self,
        args,
        logger, 
        loader: InferenceDataLoader,
        model_checkpoint: str,
        model_class=None,
        device: str = 'cuda',
        
    ):
        """
        Initialize the hybrid NSM inference interface.
        
        Args:
            args: Configuration dictionary for model initialization
            logger: Logger for logging information
            model_checkpoint: Path to the trained model checkpoint (.ckpt file)
            data_folder: Path to folder containing metadata files
                (entities.txt, relations.txt, vocab_new.txt, word_emb_300d.npy, etc.)
            model_class: The model class to use (will import if None)
            device: Device for inference ('cuda' or 'cpu')
        
        Raises:
            FileNotFoundError: If checkpoint or metadata files are not found
        """
        self.logger = logger

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.loader = loader
        # Load metadata 
        # print("model_checkpoint: ", model_checkpoint)
        # Load model
        self.model = self._load_model(args, logger, model_checkpoint, model_class)
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(
        self,
        args,
        logger,
        checkpoint_path: str,
        model_class=None,
        # config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Load a trained NSM model from checkpoint.
        
        Args:
            args: Configuration dict (required if model_class is provided)
            checkpoint_path: Path to model checkpoint
            model_class: Model class (if None, will try to import from NSM.train.init)
            
            
        Returns:
            Loaded model in eval mode
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model if class is provided
        if model_class is not None and args is not None:
            model = model_class(args)
        else:
            # Try importing default model initialization
            try:
                from NSM.train.init import init_nsm
                args = args or {}
                model = init_nsm(
                    vars(args),
                    logger=logger,
                    num_entity=len(self.loader.entity2id),
                    num_relation=self.loader.num_kb_relation,
                    num_word=len(self.loader.word2id)
                )
            except ImportError as e:
                raise ImportError(
                    f"Could not load model. Please provide model_class. Error: {e}"
                )
        
        # Load state dict
        core_model = model.model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            core_model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            core_model.model.load_state_dict(checkpoint['state_dict'])
        else:
            core_model.load_state_dict(checkpoint)
        
        print(f"Loaded model from {checkpoint_path}")
        return model
    
    def register_subgraph(
        self,
        subgraph_id: str,
        subgraph_edges: List[List[int]],
        local_entities: List[int],
        query_entity_ids: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a pre-computed subgraph for later use in inference.
        
        Args:
            subgraph_id: Unique identifier for this subgraph
            subgraph_edges: List of [head, relation, tail] triples (as entity/relation indices)
            local_entities: List of entity IDs to consider in this subgraph
            query_entity_ids: Query entity IDs (seed entities)
            metadata: Optional metadata dictionary associated with this subgraph
        """
        self.subgraph_store[subgraph_id] = {
            'edges': subgraph_edges,
            'local_entities': local_entities,
            'query_entities': query_entity_ids or [],
            'metadata': metadata or {}
        }
        print(f"Registered subgraph {subgraph_id} with {len(local_entities)} entities")
    
    def load_subgraph_batch(
        self,
        batch_file: str
    ) -> None:
        """
        Load a batch of subgraphs from a JSON file.
        
        The file should contain a list of dictionaries, each with:
        - 'id': subgraph ID
        - 'subgraph': {'tuples': [[h, r, t], ...]}
        - 'entities': [entity_id, ...] (local entities in subgraph)
        - 'entities' field from data (query entity IDs)
        
        Args:
            batch_file: Path to JSON file containing subgraph batch
        """
        with open(batch_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                subgraph_id = data.get('id', None)
                if subgraph_id is None:
                    continue
                
                edges = data['subgraph']['tuples']
                local_entities = data.get('entities', [])
                query_entities = data.get('entities', [])  # Query entities
                
                self.register_subgraph(
                    subgraph_id=subgraph_id,
                    subgraph_edges=edges,
                    local_entities=local_entities,
                    query_entity_ids=query_entities,
                    metadata={'question': data.get('question', '')}
                )
    
    def tokenize_query(self,question_text):
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        for w_idx, w in enumerate(question_text.split(' ')):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w == '':
                continue
            words += [w]
        return words
    
    def _build_adjacency_matrix(
        self,
        edges: List[List[int]],
        local_entities: List[int]
    ) -> np.ndarray:
        """
        Build adjacency matrix representation from edge list.
        
        Args:
            edges: List of [head, relation, tail] triples
            local_entities: List of local entity indices for mapping
            
        Returns:
            Adjacency matrix representation
        """
        # Create entity mapping (global to local)
        entity_to_local = {ent: idx for idx, ent in enumerate(local_entities)}
        
        # Build sparse adjacency representation
        # This should match the model's expected input format
        # Typically stored as (head_list, rel_list, tail_list)
        head_list = []
        rel_list = []
        tail_list = []
        
        for head, rel, tail in edges:
            if head in entity_to_local and tail in entity_to_local:
                head_list.append(entity_to_local[head])
                rel_list.append(rel)
                tail_list.append(entity_to_local[tail])
        
        return (head_list, rel_list, tail_list)
    
    def infer(
        self,
        subgraph_id: str,
        query: str,
        top_k: int = 10,
        seed_entity_id: int = 0,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Run inference on a query given a stored subgraph.
        
        Args:
            subgraph_id: ID of pre-registered subgraph to use
            query: Natural language query string
            top_k: Number of top answers to return
            threshold: Probability threshold for filtering candidates
            
        Returns:
            Dictionary containing:
            - 'top_k_answers': List of top-k answer entities (as names)
            - 'top_k_ids': List of top-k answer entity IDs
            - 'top_k_scores': List of confidence scores for top-k answers
            - 'all_candidates': All candidates above threshold with scores
            - 'query': Original query string
            - 'subgraph_id': The subgraph ID used
            - 'metadata': Associated metadata
            
        Raises:
            KeyError: If subgraph_id is not registered
            RuntimeError: If model is still in training mode
        """
        
        # Check if subgraph is registered
        # if subgraph_id not in self.subgraph_store:
        #     raise KeyError(f"Subgraph {subgraph_id} not registered. Available: {list(self.subgraph_store.keys())}")
        
        if self.model.training:
            self.model.eval()
            print("Warning: Model was in training mode, switching to eval")
        
        batch = self.loader.get_batch_for_inference(
            subgraph_id=subgraph_id,
            question_text=query,
            seed_entity_id=seed_entity_id,
            return_dict=False  # Returns tuple format
        )
        
        # Move batch to device
        # batch = self._batch_to_device(batch)
        local_entities = batch[0]
        query_entities = batch[1]
        # Run inference
        with torch.no_grad():
            loss, extras, pred_dist, tp_list = self.model(batch)
        
        # Process results
        results = self._process_predictions(
            pred_dist,
            local_entities.squeeze().tolist(),
            query_entities.squeeze().tolist(),
            top_k,
            # threshold
        )
        
        # Add metadata
        results.update({
            'query': query,
            'subgraph_id': subgraph_id,
            # 'metadata': subgraph_data['metadata']
        })
        
        return results
    
    def _prepare_batch(
        self,
        local_entities: List[int],
        query_entities: List[int],
        query_tokens: List[int],
        adj_mat: Tuple[List, List, List]
    ) -> Tuple:
        """
        Prepare batch data for model inference.
        
        Returns:
            Tuple in format expected by NSM model
        """
        # This format should match what the model expects
        # Placeholder implementation - adjust based on actual model requirements
        
        # Convert to numpy/torch tensors
        local_entities_arr = np.array(local_entities, dtype=int)
        query_entities_arr = np.zeros(len(local_entities), dtype=float)
        
        # Mark query entities
        for qe in query_entities:
            if qe in local_entities:
                idx = local_entities.index(qe)
                query_entities_arr[idx] = 1.0
        
        # Build fact matrix (sparse representation)
        fact_mat = np.zeros((len(local_entities), len(local_entities)), dtype=float)
        head_list, rel_list, tail_list = adj_mat
        for h, r, t in zip(head_list, rel_list, tail_list):
            fact_mat[h, t] += 1.0  # Can be weighted by relation
        
        # Pad query tokens
        max_query_len = 20  # Adjust as needed
        query_tokens_padded = query_tokens[:max_query_len] + [len(self.word2id)] * (max_query_len - len(query_tokens))
        
        # Create seed distribution
        seed_dist = query_entities_arr.copy()
        dummy_array = np.random.randn(1, len(local_entities))
        # Return batch tuple
        return (
            local_entities_arr,  # local_entity
            query_entities_arr,  # query_entities
            fact_mat,  # kb_adj_mat (fact matrix)
            np.array(query_tokens_padded),  # query_text
            seed_dist,  # seed_dist
            None,  # true_batch_id
            dummy_array, # answer_dist (not used in inference)
        )
    
    def _batch_to_device(self, batch: Tuple) -> Tuple:
        """Convert batch to specified device (cuda/cpu)."""
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dists = batch
        
        local_entity = local_entity[np.newaxis, :]
        query_entities = query_entities[np.newaxis, :]
        kb_adj_mat = kb_adj_mat[np.newaxis, :]
        query_text = query_text[np.newaxis, :]
        seed_dist = seed_dist[np.newaxis, :]
        answer_dists = answer_dists[np.newaxis, :]
        
        return (local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dists)
    
    def _process_predictions(
        self,
        pred_dist: torch.Tensor,
        local_entities: List[int],
        query_entities: List[int],
        top_k: int,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Process model predictions and return top-k answers.
        """
        # 1. Get the scores for the first sample in the batch
        # Ensure it's a 1D tensor for processing
        scores = pred_dist[0] 
        
        # 2. Use torch.topk to get values and indices on the GPU/CPU efficiently
        # We take more than top_k initially because we might filter out query_entities
        k_search = min(len(local_entities), top_k + len(query_entities))
        top_scores, top_indices = torch.topk(scores, k=k_search, largest=True)
        
        # Move to CPU and convert to list for iteration
        top_scores = top_scores.cpu().tolist()
        top_indices = top_indices.cpu().tolist()
        
        candidates = []
        query_set = set(query_entities)
        
        # 3. Map indices back to entity IDs and filter
        for idx, score in zip(top_indices, top_scores):
            if score < threshold:
                continue
                
            entity_id = local_entities[idx]
            
            if entity_id in query_set:
                continue
                
            candidates.append((entity_id, score))
            
            # Stop once we have reached the desired top_k
            if len(candidates) >= top_k:
                break
        
        # 4. Prepare output lists
        top_k_ids = [ent_id for ent_id, _ in candidates]
        top_k_scores = [score for _, score in candidates]
        
        return {
            'top_k_ids': top_k_ids,
            'top_k_scores': top_k_scores,
            'num_candidates': len(candidates)
        }
    
    def batch_infer(
        self,
        queries: Dict[str, str],
        top_k: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run inference on multiple queries.
        
        Args:
            queries: Dictionary mapping {subgraph_id: query_text}
            top_k: Number of top answers to return per query
            
        Returns:
            Dictionary mapping subgraph_id to results
        """
        results = {}
        for subgraph_id, query in queries.items():
            try:
                results[subgraph_id] = self.infer(subgraph_id, query, top_k)
            except Exception as e:
                print(f"Error inferring for subgraph {subgraph_id}: {e}")
                results[subgraph_id] = {'error': str(e)}
        
        return results

