# Hybrid Inference Interface Overview

## Tổng Quan
Hai file này cung cấp interface hoàn chỉnh để chạy inference trên NSM model trong setting hybrid:
- **hybrid_inference_interface.py**: Interface chính cho inference
- **inference_loader.py**: Data loader chuẩn bị dữ liệu cho inference

---

## 1. hybrid_inference_interface.py

### Lớp chính: `HybridNSMInference`

**Mục đích**: Interface giao tiếp với model NSM được huấn luyện, xử lý inference trên subgraph đã tính toán sẵn.

**Các tính năng chính**:

- **Khởi tạo model**: Tải checkpoint đã huấn luyện, setup device (cuda/cpu)
- **Quản lý subgraph**: Đăng ký hoặc tải batch subgraph từ file JSON
- **Inference**: Nhận input (subgraph_id, query, seed_entity) → trả về top-k answers
- **Xử lý query**: Tokenize text, encode words thành indices
- **Xây dựng adjacency matrix**: Chuyển đổi edge list sang format ma trận cho model

**Phương thức chính**:
- `__init__()`: Khởi tạo với model checkpoint, logger, data loader
- `_load_model()`: Tải model từ checkpoint file
- `register_subgraph()`: Đăng ký một subgraph cụ thể
- `load_subgraph_batch()`: Tải batch subgraph từ file JSON lines
- `infer()`: Chạy inference trên query + subgraph → trả về top-k answers với scores

**Output inference**:
```
{
  'top_k_answers': [entity_names],
  'top_k_ids': [entity_ids],
  'top_k_scores': [scores],
  'subgraph_id': str,
  'metadata': dict
}
```

---

## 2. inference_loader.py

### Lớp chính: `InferenceDataLoader`

**Mục đích**: Data loader chuyên biệt cho phase inference, chuẩn bị dữ liệu theo định dạng compatible với model.

**Các tính năng chính**:

- **Tải subgraph**: Load pre-computed subgraph từ JSON lines file
- **Xây dựng mapping**: Global → Local entity mapping (trong context của subgraph)
- **Encode question**: Chuyển text query thành word indices
- **Xây dựng fact matrix**: Chuyển edge tuples sang format (head_list, rel_list, tail_list) với inverse relations & self-loops nếu cần
- **Chuẩn bị batch**: Tạo batch format compatible với model's forward pass

**Phương thức chính**:
- `__init__()`: Khởi tạo với config, word2id, relation2id, entity2id mappings
- `_load_subgraphs()`: Tải subgraph từ file
- `get_batch_for_inference()`: Chuẩn bị batch cho một query inference
- `_encode_question()`: Convert question text → word ID array
- `_build_fact_mat()`: Xây dựng fact matrix từ subgraph tuples (hỗ trợ inverse relations, self-loops)
- `_build_global2local_map()`: Tạo mapping entity global IDs → local IDs trong subgraph

**Input `get_batch_for_inference()`**:
- `subgraph_id`: ID của subgraph cần dùng
- `question_text`: Query text
- `seed_entity_id`: Seed entity global ID

**Output**:
```
Tuple: (candidate_entities, query_entities, kb_adj_mat, query_text, 
        seed_dist, true_batch_id, answer_dist)
```
Compatible trực tiếp với model forward pass.

---

## Mối liên hệ giữa hai file

1. **HybridNSMInference** sử dụng **InferenceDataLoader** để chuẩn bị dữ liệu
2. InferenceDataLoader chuẩn bị batch format → HybridNSMInference đưa vào model
3. Model tính toán → HybridNSMInference xử lý output trả về kết quả cuối cùng

## Workflow Inference

```
Query + Subgraph ID
       ↓
InferenceDataLoader.get_batch_for_inference()
       ↓
Model forward pass
       ↓
HybridNSMInference._process_predictions()
       ↓
Return top-k answers
```
