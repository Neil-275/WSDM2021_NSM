"""
Hybrid NSM Inference Interface - Complete Package

This directory contains a complete, production-ready interface for the NSM hybrid model
inference system. Use this package for KBQA inference on pre-computed subgraphs.

Files Overview:
===============

1. CORE INTERFACE
   - hybrid_inference_interface.py
     The main HybridNSMInference class implementing all inference logic
     Size: ~600 lines
     Key features:
       * Single and batch inference
       * Subgraph registration and loading
       * Metadata management
       * Query tokenization
       * Top-k answer extraction

2. CONFIGURATION & SETUP
   - hybrid_inference_config.py
     Pre-built configurations for common use cases
     Includes: CWQ, WebQSP, Production, Development, CPU-only, Batch, etc.
     Builder pattern for custom configurations

3. EXAMPLES & GUIDES
   - example_hybrid_inference.py
     Comprehensive examples demonstrating all features
     6 detailed examples with explanations

   - quick_start_hybrid_inference.py
     Quick start guide with troubleshooting
     Common tasks, performance tips, and troubleshooting

4. DOCUMENTATION
   - HYBRID_INFERENCE_README.md
     Complete documentation with API reference
     Architecture, usage patterns, advanced features

5. THIS FILE
   - __init__.py (this file)
     Package overview and quick reference


Quick Reference
===============

Initialize:
-----------
from hybrid_inference_interface import HybridNSMInference

inferencer = HybridNSMInference(
    model_checkpoint='checkpoint/model.ckpt',
    data_folder='data/',
    device='cuda'
)

Load subgraphs:
---------------
inferencer.load_subgraph_batch('data/test.json')
# or manually:
inferencer.register_subgraph(
    subgraph_id='0',
    subgraph_edges=[[0, 1, 2]],
    local_entities=[0, 1, 2]
)

Run inference:
--------------
results = inferencer.infer(
    subgraph_id='0',
    query='What is the capital of France?',
    top_k=10
)

Access results:
---------------
results['top_k_answers']      # List of entity names
results['top_k_ids']          # List of entity IDs
results['top_k_scores']       # List of confidence scores
results['num_candidates']     # Total candidates above threshold
results['query']              # Original query
results['subgraph_id']        # Subgraph ID used


Configuration Templates
=======================

from hybrid_inference_config import (
    setup_cwq_inferencer,           # CWQ Benchmark
    setup_webqsp_inferencer,        # WebQSP Benchmark
    setup_production_inferencer,    # Production setup
    setup_dev_inferencer,           # Development
    setup_cpu_inferencer,           # CPU-only
    InferencerBuilder,              # Custom builder
)

# Quick setup
inferencer = setup_cwq_inferencer()

# Or build custom
inferencer = (InferencerBuilder()
    .set_checkpoint('model.ckpt')
    .set_data_folder('data/')
    .set_device('cuda')
    .set_top_k(10)
    .build()
)


API Overview
============

HybridNSMInference Class Methods:

Initialization & Loading:
  - __init__()                Initialization with model and metadata
  - _load_metadata()          Load all metadata dictionaries
  - _load_model()             Load trained model checkpoint

Subgraph Management:
  - register_subgraph()       Register a subgraph for inference
  - load_subgraph_batch()     Load multiple subgraphs from file

Inference:
  - infer()                   Single query inference
  - batch_infer()             Batch query inference
  - tokenize_query()          Tokenize query to word IDs

Internal Methods:
  - _prepare_batch()          Prepare batch data for model
  - _batch_to_device()        Move batch to device
  - _process_predictions()    Process model predictions
  - _build_adjacency_matrix() Build graph representation


Data Format
===========

Subgraph File (JSON Lines):
{
  "id": "0",
  "question": "What is...?",
  "entities": [0, 1, 2, ...],
  "subgraph": {
    "tuples": [[head, rel, tail], ...]
  }
}

Results Dictionary:
{
  'top_k_answers': ['Paris', 'France', ...],
  'top_k_ids': [123, 456, ...],
  'top_k_scores': [0.95, 0.87, ...],
  'all_candidates': [('Paris', 0.95), ...],
  'num_candidates': 47,
  'query': 'What is the capital?',
  'subgraph_id': '0',
  'metadata': {...}
}


Common Patterns
===============

Pattern 1: Single Query, Single Subgraph
-----------------------------------------
inferencer = HybridNSMInference(...)
inferencer.load_subgraph_batch('data.json')
result = inferencer.infer('0', 'query', top_k=10)
print(result['top_k_answers'][0])

Pattern 2: Multiple Queries, Multiple Subgraphs
-----------------------------------------
queries = {'0': 'query1', '1': 'query2', ...}
results = inferencer.batch_infer(queries, top_k=10)
for subgraph_id, result in results.items():
    print(f"{subgraph_id}: {result['top_k_answers'][0]}")

Pattern 3: Custom Subgraph
-----------------------------------------
inferencer.register_subgraph(
    subgraph_id='custom',
    subgraph_edges=edges,
    local_entities=entities,
    query_entity_ids=query_entities
)
result = inferencer.infer('custom', 'query')

Pattern 4: Production Server
-----------------------------------------
# Initialize once at startup
inferencer = setup_production_inferencer()
inferencer.load_subgraph_batch('data.json')

# Use in request handler
@app.route('/infer')
def infer_handler():
    data = request.json
    results = inferencer.infer(data['id'], data['query'])
    return json.dumps(results)


Installation & Dependencies
============================

Requirements:
- Python 3.8+
- PyTorch 1.9+
- NumPy
- (Optional) CUDA Toolkit for GPU inference

The interface imports from NSM module:
- from NSM.train.init import init_nsm  # For automatic model loading
- from NSM.train.trainer_nsm import Trainer_KBQA  # From existing trainer

No additional dependencies required. The interface uses only PyTorch and NumPy.


Performance Characteristics
============================

Initialization:
- Time: 2-5 seconds (load model + metadata)
- Memory: 100-500 MB (depending on model size)

Single Inference:
- GPU: 10-100 ms per query
- CPU: 100 ms - 1 sec per query
- Memory: ~50-200 MB

Batch Inference (100 queries):
- GPU: 1-10 seconds total
- CPU: 10-100 seconds total

Metadata:
- Entities: Typically 50k-1M entities
- Relations: Typically 50-1000 relations
- Vocabulary: Typically 20k-100k words


Tests & Validation
==================

Quick validation:
python quick_start_hybrid_inference.py

Full examples:
python example_hybrid_inference.py

Configuration verification:
python hybrid_inference_config.py


Troubleshooting
===============

See quick_start_hybrid_inference.py for:
- FileNotFoundError resolution
- KeyError debugging
- CUDA memory management
- Slow inference fixes
- Model configuration issues

See HYBRID_INFERENCE_README.md for:
- Complete API reference
- Advanced features
- Integration examples
- Performance tuning


Usage Examples
==============

Example 1: Basic Inference
-----------
from hybrid_inference_interface import HybridNSMInference
from hybrid_inference_config import setup_cwq_inferencer

# Initialize
inferencer = setup_cwq_inferencer()
inferencer.load_subgraph_batch('data/cwq/test.json')

# Infer
results = inferencer.infer(
    subgraph_id='0',
    query='What is the capital of France?',
    top_k=10
)

print(results['top_k_answers'])  # ['Paris', ...]

Example 2: Batch Processing
-----------
queries = {
    '0': 'Query 1',
    '1': 'Query 2',
    '2': 'Query 3',
}

results = inferencer.batch_infer(queries, top_k=5)

for subgraph_id, result in results.items():
    print(f"{subgraph_id}: {result['top_k_answers'][0]}")

Example 3: Custom Configuration
-----------
from hybrid_inference_config import InferencerBuilder

inferencer = (InferencerBuilder()
    .set_checkpoint('my_model.ckpt')
    .set_data_folder('my_data/')
    .set_device('cuda')
    .set_top_k(15)
    .set_threshold(0.1)
    .build()
)

Example 4: Integration with Trainer
-----------
from NSM.train.trainer_nsm import Trainer_KBQA
from hybrid_inference_interface import HybridNSMInference

trainer = Trainer_KBQA(args, logger)
inferencer = HybridNSMInference(
    model_checkpoint='checkpoint.ckpt',
    data_folder=args['data_folder'],
    device='cuda'
)


File Structure
==============

WSDM2021_NSM/
├── hybrid_inference_interface.py    # Main interface class
├── hybrid_inference_config.py        # Configuration templates
├── example_hybrid_inference.py       # Usage examples
├── quick_start_hybrid_inference.py   # Quick start guide
├── HYBRID_INFERENCE_README.md        # Complete documentation
├── __init__.py                       # This file
└── README.md                         # (Original NSM README)
    ├── infer_nsm.py                  # (Original inference)
    ├── main_nsm.py                   # (Original training)
    └── ...                           # (Other NSM files)


Key Design Decisions
====================

1. Clean Interface: Single HybridNSMInference class handles all operations
2. No Global State: Can create multiple inferencer instances
3. Flexible Input: Supports file loading, manual registration, batch processing
4. Device Agnostic: Works on CPU/GPU with automatic fallback
5. Minimal Dependencies: Uses only PyTorch and NumPy
6. Type Hints: Full type hints for IDE support
7. Comprehensive Docs: Extensive documentation and examples
8. Error Handling: Clear error messages with suggestions
9. Configuration: Pre-built configs for common scenarios
10. Extensibility: Easy to extend with custom features


Integration Checklist
=====================

Before using in production:

☐ Verify model checkpoint path exists
☐ Verify data folder contains all metadata files
☐ Test on development data first
☐ Benchmark inference speed on target hardware
☐ Verify GPU memory is sufficient
☐ Set up proper error handling in your app
☐ Load subgraphs once at startup
☐ Use batch_infer for multiple queries
☐ Monitor inference latency in production
☐ Set up logging and monitoring


Support & Future Enhancements
=============================

Current Features:
✓ Single and batch inference
✓ Subgraph loading from files
✓ Manual subgraph registration
✓ Customizable thresholds
✓ GPU/CPU support
✓ Multi-model ensemble (basic)

Potential Future Enhancements:
- Asynchronous inference
- Multi-GPU support
- Model quantization
- Caching layer
- Distributed inference
- REST API wrapper
- WebSocket support
- Advanced ensemble methods
- Uncertainty estimation
- Explainability features


License & Attribution
=====================

This interface is built on top of the NSM KBQA model.
See the original README.md for attribution and license information.


Questions & Support
===================

For issues with the interface:
1. Check HYBRID_INFERENCE_README.md for complete documentation
2. See quick_start_hybrid_inference.py for troubleshooting
3. Review example_hybrid_inference.py for usage patterns
4. Check hybrid_inference_config.py for configuration options


Version
=======

Interface Version: 1.0
Release Date: April 2026
Tested with: PyTorch 1.9+, Python 3.8+, CUDA 11.0+

"""

# Import main class for easier access
from hybrid_inference_interface import HybridNSMInference

__all__ = ['HybridNSMInference']

__version__ = '1.0'
__author__ = 'KBQA Team'
__description__ = 'Hybrid NSM Inference Interface for KBQA'
