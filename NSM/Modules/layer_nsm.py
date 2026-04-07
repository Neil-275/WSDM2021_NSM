import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import math
import time
VERY_NEG_NUMBER = -100000000000
VERY_SMALL_NUMBER = 1e-10


class TypeLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, linear_drop, device):
        super(TypeLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        # self.kb_head_linear = nn.Linear(in_features, out_features)
        self.kb_self_linear = nn.Linear(in_features, out_features)
        # Initialize weights with Xavier uniform for numerical stability
        nn.init.xavier_uniform_(self.kb_self_linear.weight)
        nn.init.zeros_(self.kb_self_linear.bias)
        # self.kb_tail_linear = nn.Linear(out_features, out_features)
        self.device = device

    def forward(self, local_entity, edge_list, rel_features):
        '''
        input_vector: (batch_size, max_local_entity)
        curr_dist: (batch_size, max_local_entity)
        instruction: (batch_size, hidden_size)
        '''
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = edge_list
        num_fact = len(fact_ids)
        batch_size, max_local_entity = local_entity.size()
        hidden_size = self.in_features
        # print("Input types - batch_heads:", type(batch_heads), "fact_ids:", type(fact_ids))
        fact2head = torch.LongTensor(np.array([batch_heads, fact_ids])).to(self.device)
        fact2tail = torch.LongTensor(np.array([batch_tails, fact_ids])).to(self.device)
        batch_rels = torch.LongTensor(batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        val_one = torch.ones_like(batch_ids).float().to(self.device)

        # print("Prepare data:{:.4f}".format(time.time() - st))
        # Step 1: Calculate value for every fact with rel and head
        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))
        fact_val = self.kb_self_linear(fact_rel)
        # fact_val = self.kb_self_linear(fact_rel)#self.kb_head_linear(self.linear_drop(fact_ent))
        
        # Clamp to prevent numerical instability
        fact_val = torch.clamp(fact_val, min=-1e6, max=1e6)

        # Step 3: Edge Aggregation with Sparse MM
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))

        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))
        tail_repr = torch.sparse.mm(fact2tail_mat, fact_val)
        head_repr = torch.sparse.mm(fact2head_mat, fact_val)
        
        # Handle potential NaN values from sparse operations
        tail_repr = torch.nan_to_num(tail_repr, nan=0.0, posinf=1e6, neginf=-1e6)
        head_repr = torch.nan_to_num(head_repr, nan=0.0, posinf=1e6, neginf=-1e6)
        
        f2e_emb = F.relu(tail_repr + head_repr)
        
        # Final safety check and clamp
        f2e_emb = torch.nan_to_num(f2e_emb, nan=0.0, posinf=1e6, neginf=-1e6)
        assert not torch.isnan(f2e_emb).any(), f"NaN detected in f2e_emb. tail_repr has NaN: {torch.isnan(tail_repr).any()}, head_repr has NaN: {torch.isnan(head_repr).any()}"

        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)

        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        # return torch.sparse.FloatTensor(indices, values, size).to(self.device)
        return torch.sparse_coo_tensor(indices, values, size, device=self.device)


class STLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, linear_drop, device):
        super(STLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        # self.kb_head_linear = nn.Linear(in_features, out_features)
        self.kb_self_linear = nn.Linear(in_features, out_features)
        # Initialize weights with Xavier uniform for numerical stability
        # nn.init.xavier_uniform_(self.kb_self_linear.weight)
        # nn.init.zeros_(self.kb_self_linear.bias)
        # self.kb_tail_linear = nn.Linear(out_features, out_features)
        self.device = device

    def forward(self, input_vector, edge_list, curr_dist, instruction, rel_features):
        '''
        input_vector: (batch_size, max_local_entity, hidden_size)
        curr_dist: (batch_size, max_local_entity)
        instruction: (batch_size, hidden_size)
        '''
        print("123")
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = edge_list
        num_fact = len(fact_ids)
        batch_size, max_local_entity, hidden_size = input_vector.size()
        # input_vector = input_vector.view(batch_size * max_local_entity, hidden_size)
        # fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        head2fact = torch.LongTensor([fact_ids, batch_heads]).to(self.device)
        # batch_heads = torch.LongTensor(batch_heads).to(self.device)
        # batch_tails = torch.LongTensor(batch_tails).to(self.device)
        batch_rels = torch.LongTensor(batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        val_one = torch.ones_like(batch_ids).float().to(self.device)

        # print("Prepare data:{:.4f}".format(time.time() - st))
        # Step 1: Calculate value for every fact with rel and head
        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=batch_ids)
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))
        fact_val = F.relu(self.kb_self_linear(fact_rel) * fact_query)
        # Clamp to prevent numerical instability
        fact_val = torch.clamp(fact_val, min=-1e6, max=1e6)
        # fact_val = self.kb_self_linear(fact_rel)#self.kb_head_linear(self.linear_drop(fact_ent))

        # Step 3: Edge Aggregation with Sparse MM
        head2fact_mat = self._build_sparse_tensor(head2fact, val_one, (num_fact, batch_size * max_local_entity))
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact_prior = torch.sparse.mm(head2fact_mat, curr_dist.view(-1, 1))
        # (num_fact, batch_size * max_local_entity) (batch_size * max_local_entity, 1) -> (num_fact, 1)
        
        # Handle potential NaN values from sparse operations
        fact_prior = torch.nan_to_num(fact_prior, nan=0.0, posinf=1e6, neginf=-1e6)

        # fact_val = fact_val * edge_e
        fact_val = fact_val * fact_prior
        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))
        f2e_emb = torch.sparse.mm(fact2tail_mat, fact_val)
        
        # Handle potential NaN values
        f2e_emb = torch.nan_to_num(f2e_emb, nan=0.0, posinf=1e6, neginf=-1e6)
        assert not torch.isnan(f2e_emb).any(), f"NaN detected in STLayer f2e_emb"

        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)

        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        # return torch.sparse.FloatTensor(indices, values, size).to(self.device)
        return torch.sparse_coo_tensor(indices, values, size, device=self.device)