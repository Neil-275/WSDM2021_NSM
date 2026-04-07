import json
import numpy as np
import re
import os
from collections import Counter
from NSM.data.basic_dataset import BasicDataLoader


class InferenceDataLoader(object):
    """
    Data loader for inference-mode NSM model.
    
    This class is specifically designed for inference where:
    - No ground-truth answers are provided
    - Subgraphs are pre-computed and loaded from files
    - Users provide query ID, query text, and seed entity for each inference call
    
    The class returns data in the same format as BasicDataLoader/SingleDataLoader
    to be compatible with the model's forward pass.
    """
    
    def __init__(self, config, word2id, relation2id, entity2id):
        """
        Initialize the inference data loader.
        
        Args:
            config (dict): Configuration dictionary containing:
                - 'use_inverse_relation': bool, whether to use inverse relations
                - 'use_self_loop': bool, whether to use self-loop edges
                - 'num_step': int, number of reasoning steps
                - 'subgraph_file': str, path to pre-computed subgraphs (JSON lines format)
            word2id (dict): Mapping from words to IDs
            relation2id (dict): Mapping from relation names to IDs
            entity2id (dict): Mapping from entity names/IDs to global entity IDs
        """
        self._parse_args(config, word2id, relation2id, entity2id)
        self._load_subgraphs(config)
    
    def _parse_args(self, config, word2id, relation2id, entity2id):
        """Parse configuration and build mappings."""
        self.use_inverse_relation = config.use_inverse_relation
        self.use_self_loop = config.use_self_loop
        self.num_step = config.num_step
        self.max_local_entity = 0
        self.max_query_word = 0
        self.max_facts = 0
        
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2entity = {i: entity for entity, i in entity2id.items()}
        
        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        
        print("Inference DataLoader initialized:")
        print("  Entity vocab size: {}".format(len(entity2id)))
        print("  Relation vocab size: {} (KB relations in use: {})".format(
            len(relation2id), self.num_kb_relation))
        print("  Word vocab size: {}".format(len(word2id)))
    
    def _load_subgraphs(self, args):
        """
        Load pre-computed subgraphs from file.
        
        Expected format (JSON lines): Each line contains a dict with:
        {
            'id': str,  # unique identifier for the subgraph
            'subgraph': {
                'entities': [entity_id1, entity_id2, ...],
                'tuples': [[subj_id, rel_id, obj_id], ...]
            }
        }
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments containing 'subgraph_file' path
        """
        self.subgraphs = {}
        self.subgraph_ids = []
        
        subgraph_file = hasattr(args, 'subgraph_file') and args.subgraph_file or args.data_folder + 'test_simple.json'
        if subgraph_file is None:
            print("Warning: No subgraph_file provided. Subgraphs must be added manually.")
            return
        
        print(f"Loading subgraphs from {subgraph_file}...")
        with open(subgraph_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    subgraph_id = data['id']
                    subgraph = data['subgraph']
                    self.subgraphs[subgraph_id] = subgraph
                    self.subgraph_ids.append(subgraph_id)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing subgraph: {e}")
                    continue
        
        print(f"Loaded {len(self.subgraphs)} subgraphs")
    
    @staticmethod
    def tokenize_sent(question_text):
        """
        Tokenize question text.
        
        Args:
            question_text (str): Raw question text
            
        Returns:
            list: List of tokens
        """
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        for w_idx, w in enumerate(question_text.split(' ')):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w == '':
                continue
            words += [w]
        return words
    
    def _encode_question(self, question_text):
        """
        Encode question text to word IDs.
        
        Args:
            question_text (str): Raw question text
            
        Returns:
            np.ndarray: Encoded question with shape (max_query_word,)
        """
        words = self.tokenize_sent(question_text)
        max_len = max(len(words), self.max_query_word)
        if max_len > self.max_query_word:
            self.max_query_word = max_len
        
        query_text = np.full(self.max_query_word, len(self.word2id), dtype=int)
        for j, word in enumerate(words):
            if word in self.word2id:
                query_text[j] = self.word2id[word]
        return query_text
    
    def _build_global2local_map(self, subgraph):
        """
        Build mapping from global entity IDs to local (subgraph-local) entity IDs.
        
        Args:
            subgraph (dict): Subgraph dict with 'entities' key
            
        Returns:
            dict: Mapping from global entity ID to local entity ID
        """
        g2l = {}
        if 'entities' in subgraph:
            for entity_id in subgraph['entities']:
                if entity_id not in g2l:
                    g2l[entity_id] = len(g2l)
        return g2l
    
    def _build_fact_mat(self, subgraph, g2l):
        """
        Build fact matrix from subgraph tuples.
        
        Args:
            subgraph (dict): Subgraph containing 'tuples' key
            g2l (dict): Global-to-local entity mapping
            
        Returns:
            tuple: (head_list, rel_list, tail_list) as numpy arrays
        """
        head_list = []
        rel_list = []
        tail_list = []
        
        if 'tuples' not in subgraph:
            return (np.array(head_list, dtype=int),
                    np.array(rel_list, dtype=int),
                    np.array(tail_list, dtype=int))
        
        for sbj, rel, obj in subgraph['tuples']:
            if sbj in g2l and obj in g2l:
                head = g2l[sbj]
                tail = g2l[obj]
                rel_id = int(rel)
                
                head_list.append(head)
                rel_list.append(rel_id)
                tail_list.append(tail)
                
                # Add inverse relation if needed
                if self.use_inverse_relation:
                    head_list.append(tail)
                    rel_list.append(rel_id + len(self.relation2id))
                    tail_list.append(head)
        
        # Add self-loops if needed
        if self.use_self_loop:
            for local_ent in range(len(g2l)):
                head_list.append(local_ent)
                rel_list.append(self.num_kb_relation - 1)  # self-loop relation ID
                tail_list.append(local_ent)
        
        return (np.array(head_list, dtype=int),
                np.array(rel_list, dtype=int),
                np.array(tail_list, dtype=int))
    
    def get_batch_for_inference(self, subgraph_id, question_text, seed_entity_id, 
                                fact_dropout=0.0, return_dict=False):
        """
        Prepare a batch of data for a single inference query.
        
        This method retrieves a pre-computed subgraph and prepares it along with
        the query and seed entity for the model's forward pass.
        
        Args:
            subgraph_id (str): ID of the subgraph to use
            question_text (str): Question/query text
            seed_entity_id (int): Global ID of the seed entity
            fact_dropout (float): Dropout rate for facts (default: 0.0 for inference)
            return_dict (bool): If True, return a dictionary with metadata.
                               If False, return tuple format for direct model input.
            
        Returns:
            If return_dict=False (default):
                tuple: (candidate_entities, query_entities, kb_adj_mat, query_text, 
                        seed_dist, true_batch_id, answer_dist) - compatible with model forward pass
            
            If return_dict=True:
                dict: Contains above plus metadata:
                    - 'batch': The tuple above
                    - 'global2local_map': Mapping from global to local entity IDs
                    - 'local_entity_count': Number of entities in subgraph
                    - 'subgraph_id': The ID of the used subgraph
                    - 'seed_entity_local_id': Local ID of seed entity
                    - 'seed_entity_global_id': Global ID of seed entity
                    - 'answer_dist': Dummy answer distribution (zeros)
        """
        if subgraph_id not in self.subgraphs:
            raise ValueError(f"Subgraph ID '{subgraph_id}' not found. "
                           f"Available IDs: {self.subgraph_ids}")
        
        subgraph = self.subgraphs[subgraph_id]
        
        # Build global-to-local mapping
        g2l = self._build_global2local_map(subgraph)
        num_local_entity = len(g2l)
        
        if seed_entity_id not in g2l:
            raise ValueError(f"Seed entity {seed_entity_id} not in subgraph {subgraph_id}")
        
        local_seed_entity = g2l[seed_entity_id]
        
        # Initialize arrays
        candidate_entities = np.full(num_local_entity, len(self.entity2id), dtype=int)
        query_entities = np.zeros(num_local_entity, dtype=float)
        seed_distribution = np.zeros(num_local_entity, dtype=float)
        
        # Mark seed entity
        query_entities[local_seed_entity] = 1.0
        seed_distribution[local_seed_entity] = 1.0
        
        # Set candidate entities (all entities in subgraph)
        for global_ent, local_ent in g2l.items():
            candidate_entities[local_ent] = global_ent
        
        # Encode question
        query_text = self._encode_question(question_text)
        
        # Build fact matrix (with padded batch dimension for compatibility)
        kb_adj_mat = self._build_fact_mat(subgraph, g2l)
        
        # Build fact matrix compatible with model input (with batch offset)
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = \
            self._build_batched_fact_mat([kb_adj_mat], [num_local_entity], fact_dropout=fact_dropout)
        
        # Model expects true_batch_id as None for single inference
        true_batch_id = None
        
        # Dummy answer distribution (zeros) for compatibility with legacy model code
        # In inference, we don't have ground truth answers, so use zero distribution
        answer_dist = np.zeros(num_local_entity, dtype=float)
        
        # Prepare tuple format for model input (compatible with BasicDataLoader.get_batch)
        batch_tuple = (
            np.expand_dims(candidate_entities, axis=0),  # (1, num_entities)
            np.expand_dims(query_entities, axis=0),      # (1, num_entities)
            (batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list),
            np.expand_dims(query_text, axis=0),          # (1, max_query_word)
            np.expand_dims(seed_distribution, axis=0),   # (1, num_entities)
            true_batch_id,
            np.expand_dims(answer_dist, axis=0)          # (1, num_entities) - dummy for compatibility
        )
        
        if return_dict:
            return {
                'batch': batch_tuple,
                'candidate_entities': batch_tuple[0],
                'query_entities': batch_tuple[1],
                'kb_adj_mat': batch_tuple[2],
                'query_text': batch_tuple[3],
                'seed_dist': batch_tuple[4],
                'true_batch_id': batch_tuple[5],
                'answer_dist': batch_tuple[6],
                'global2local_map': g2l,
                'local_entity_count': num_local_entity,
                'subgraph_id': subgraph_id,
                'seed_entity_local_id': local_seed_entity,
                'seed_entity_global_id': seed_entity_id
            }
        else:
            return batch_tuple
    
    def _build_batched_fact_mat(self, kb_adj_mats, entity_counts, fact_dropout=0.0):
        """
        Build batched fact matrix (compatible with BasicDataLoader format).
        
        Args:
            kb_adj_mats (list): List of (head_list, rel_list, tail_list) tuples
            entity_counts (list): Number of entities in each sample
            fact_dropout (float): Dropout rate for facts
            
        Returns:
            tuple: (batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list)
        """
        batch_heads = np.array([], dtype=int)
        batch_rels = np.array([], dtype=int)
        batch_tails = np.array([], dtype=int)
        batch_ids = np.array([], dtype=int)
        
        for i, (head_list, rel_list, tail_list) in enumerate(kb_adj_mats):
            index_bias = i * entity_counts[i]
            num_fact = len(head_list)
            
            if num_fact == 0:
                continue
            
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]
            
            real_head_list = head_list[mask_index] + index_bias
            real_tail_list = tail_list[mask_index] + index_bias
            real_rel_list = rel_list[mask_index]
            
            batch_heads = np.append(batch_heads, real_head_list)
            batch_rels = np.append(batch_rels, real_rel_list)
            batch_tails = np.append(batch_tails, real_tail_list)
            batch_ids = np.append(batch_ids, np.full(len(mask_index), i, dtype=int))
        
        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        head_count = Counter(batch_heads)
        weight_list = [1.0 / head_count[head] if head in head_count else 1.0 for head in batch_heads]
        
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list
    
    def add_subgraph(self, subgraph_id, subgraph):
        """
        Manually add a subgraph to the loader.
        
        Useful for adding subgraphs programmatically instead of loading from file.
        
        Args:
            subgraph_id (str): Unique identifier for the subgraph
            subgraph (dict): Subgraph dict with 'entities' and 'tuples' keys
        """
        self.subgraphs[subgraph_id] = subgraph
        if subgraph_id not in self.subgraph_ids:
            self.subgraph_ids.append(subgraph_id)
    
    def list_available_subgraphs(self):
        """
        Get list of available subgraph IDs.
        
        Returns:
            list: Available subgraph IDs
        """
        return self.subgraph_ids
    
    def decode_text(self, np_array_x):
        """
        Decode encoded question back to text.
        
        Args:
            np_array_x (np.ndarray): Encoded question array
            
        Returns:
            str: Decoded question text
        """
        id2word = self.id2word
        tp_str = ""
        for i in range(len(np_array_x)):
            if np_array_x[i] in id2word:
                tp_str += id2word[np_array_x[i]] + " "
        return tp_str.strip()
