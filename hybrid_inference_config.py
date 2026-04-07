"""
Configuration Templates for Hybrid NSM Inference

This file contains pre-configured templates for different use cases.
Copy and modify as needed for your specific setup.
"""

from hybrid_inference_interface import HybridNSMInference
from typing import Dict, Any


# ============================================================================
# TEMPLATE 1: Standard CWQ Benchmark Setup
# ============================================================================

CONFIG_CWQ = {
    'name': 'CWQ Benchmark',
    'description': 'Standard configuration for ComplexWebQuestions benchmark',
    'model_checkpoint': 'checkpoint/CWQ_hybrid.ckpt',
    'data_folder': 'data/cwq/',
    'device': 'cuda',
    'inference_params': {
        'top_k': 10,
        'threshold': 0.05,
    }
}

def setup_cwq_inferencer() -> HybridNSMInference:
    """Initialize inferencer for CWQ benchmark."""
    inferencer = HybridNSMInference(
        model_checkpoint=CONFIG_CWQ['model_checkpoint'],
        data_folder=CONFIG_CWQ['data_folder'],
        device=CONFIG_CWQ['device']
    )
    return inferencer


# ============================================================================
# TEMPLATE 2: Standard WebQSP Benchmark Setup
# ============================================================================

CONFIG_WEBQSP = {
    'name': 'WebQSP Benchmark',
    'description': 'Standard configuration for WebQuestionsSP benchmark',
    'model_checkpoint': 'checkpoint/WebQSP_hybrid.ckpt',
    'data_folder': 'data/webqsp/',
    'device': 'cuda',
    'inference_params': {
        'top_k': 10,
        'threshold': 0.05,
    }
}

def setup_webqsp_inferencer() -> HybridNSMInference:
    """Initialize inferencer for WebQSP benchmark."""
    inferencer = HybridNSMInference(
        model_checkpoint=CONFIG_WEBQSP['model_checkpoint'],
        data_folder=CONFIG_WEBQSP['data_folder'],
        device=CONFIG_WEBQSP['device']
    )
    return inferencer


# ============================================================================
# TEMPLATE 3: Production High-Performance Setup
# ============================================================================

CONFIG_PRODUCTION = {
    'name': 'Production Setup',
    'description': 'Optimized for fast inference with GPU',
    'model_checkpoint': 'checkpoint/model_production.ckpt',
    'data_folder': '/data/kbqa/metadata/',
    'device': 'cuda',
    'inference_params': {
        'top_k': 5,  # Fewer top answers for speed
        'threshold': 0.1,  # Higher threshold for faster processing
    }
}

def setup_production_inferencer() -> HybridNSMInference:
    """
    Initialize high-performance inferencer for production.
    
    Optimizations:
    - Uses GPU for fast inference
    - Higher threshold to reduce top-k computation
    - Returns fewer top answers for lower latency
    """
    inferencer = HybridNSMInference(
        model_checkpoint=CONFIG_PRODUCTION['model_checkpoint'],
        data_folder=CONFIG_PRODUCTION['data_folder'],
        device=CONFIG_PRODUCTION['device']
    )
    return inferencer


# ============================================================================
# TEMPLATE 4: Research/Development Setup
# ============================================================================

CONFIG_DEV = {
    'name': 'Development Setup',
    'description': 'Comprehensive setup for research and experiments',
    'model_checkpoint': 'checkpoint/model_dev.ckpt',
    'data_folder': './data/dev/',
    'device': 'cuda',  # Use 'cpu' for debugging
    'inference_params': {
        'top_k': 20,  # Get more candidates for analysis
        'threshold': 0.0,  # Include all candidates
    },
    'options': {
        'save_debug_info': True,
        'verbose_logging': True,
    }
}

def setup_dev_inferencer() -> HybridNSMInference:
    """
    Initialize inferencer for development and research.
    
    Features:
    - Returns all candidates for detailed analysis
    - More verbose logging
    - Suitable for debugging
    """
    inferencer = HybridNSMInference(
        model_checkpoint=CONFIG_DEV['model_checkpoint'],
        data_folder=CONFIG_DEV['data_folder'],
        device=CONFIG_DEV['device']
    )
    return inferencer


# ============================================================================
# TEMPLATE 5: CPU-Only Setup (for resource-constrained environments)
# ============================================================================

CONFIG_CPU_ONLY = {
    'name': 'CPU-Only Setup',
    'description': 'For systems without GPU access',
    'model_checkpoint': 'checkpoint/model_cpu.ckpt',
    'data_folder': 'data/',
    'device': 'cpu',
    'inference_params': {
        'top_k': 5,  # Fewer answers for speed on CPU
        'threshold': 0.2,  # Higher threshold to reduce computation
    }
}

def setup_cpu_inferencer() -> HybridNSMInference:
    """Initialize inferencer for CPU-only systems."""
    inferencer = HybridNSMInference(
        model_checkpoint=CONFIG_CPU_ONLY['model_checkpoint'],
        data_folder=CONFIG_CPU_ONLY['data_folder'],
        device=CONFIG_CPU_ONLY['device']
    )
    return inferencer


# ============================================================================
# TEMPLATE 6: Large-Scale Batch Processing Setup
# ============================================================================

CONFIG_BATCH = {
    'name': 'Batch Processing Setup',
    'description': 'Optimized for processing large batches of queries',
    'model_checkpoint': 'checkpoint/model_batch.ckpt',
    'data_folder': 'data/batch/',
    'device': 'cuda',
    'batch_size': 100,  # Process many queries at once
    'inference_params': {
        'top_k': 10,
        'threshold': 0.05,
    }
}

def setup_batch_inferencer() -> HybridNSMInference:
    """Initialize inferencer for batch processing."""
    inferencer = HybridNSMInference(
        model_checkpoint=CONFIG_BATCH['model_checkpoint'],
        data_folder=CONFIG_BATCH['data_folder'],
        device=CONFIG_BATCH['device']
    )
    return inferencer


# ============================================================================
# TEMPLATE 7: Knowledge Graph Exploration Setup
# ============================================================================

CONFIG_KG_EXPLORATION = {
    'name': 'Knowledge Graph Exploration',
    'description': 'For exploring and analyzing KG structure',
    'model_checkpoint': 'checkpoint/model_exploration.ckpt',
    'data_folder': 'data/kg_analysis/',
    'device': 'cuda',
    'inference_params': {
        'top_k': 50,  # Many candidates for exploration
        'threshold': 0.001,  # Very low threshold
    }
}

def setup_kg_exploration_inferencer() -> HybridNSMInference:
    """Initialize inferencer for KG exploration."""
    inferencer = HybridNSMInference(
        model_checkpoint=CONFIG_KG_EXPLORATION['model_checkpoint'],
        data_folder=CONFIG_KG_EXPLORATION['data_folder'],
        device=CONFIG_KG_EXPLORATION['device']
    )
    return inferencer


# ============================================================================
# TEMPLATE 8: Real-time API Server Setup
# ============================================================================

CONFIG_API_SERVER = {
    'name': 'Real-time API Server',
    'description': 'For serving requests over HTTP/REST API',
    'model_checkpoint': 'checkpoint/model_api.ckpt',
    'data_folder': '/var/lib/kbqa/data/',
    'device': 'cuda',
    'inference_params': {
        'top_k': 5,
        'threshold': 0.1,
    },
    'server_config': {
        'host': '0.0.0.0',
        'port': 5000,
        'workers': 4,
        'timeout': 30,  # seconds
    }
}

def setup_api_server_inferencer() -> HybridNSMInference:
    """Initialize inferencer for API server deployment."""
    inferencer = HybridNSMInference(
        model_checkpoint=CONFIG_API_SERVER['model_checkpoint'],
        data_folder=CONFIG_API_SERVER['data_folder'],
        device=CONFIG_API_SERVER['device']
    )
    return inferencer


# ============================================================================
# TEMPLATE 9: Multi-Model Ensemble Setup
# ============================================================================

CONFIG_ENSEMBLE = {
    'name': 'Multi-Model Ensemble',
    'description': 'For combining predictions from multiple models',
    'models': [
        {
            'checkpoint': 'checkpoint/model_1.ckpt',
            'weight': 0.5,
            'name': 'Model_1'
        },
        {
            'checkpoint': 'checkpoint/model_2.ckpt',
            'weight': 0.3,
            'name': 'Model_2'
        },
        {
            'checkpoint': 'checkpoint/model_3.ckpt',
            'weight': 0.2,
            'name': 'Model_3'
        }
    ],
    'data_folder': 'data/ensemble/',
    'device': 'cuda',
    'ensemble_method': 'weighted_average',  # or 'voting', 'product', etc.
}

def setup_ensemble_inferencer() -> Dict[str, HybridNSMInference]:
    """Initialize multiple inferencers for ensemble."""
    inferencers = {}
    for model_config in CONFIG_ENSEMBLE['models']:
        name = model_config['name']
        inferencers[name] = HybridNSMInference(
            model_checkpoint=model_config['checkpoint'],
            data_folder=CONFIG_ENSEMBLE['data_folder'],
            device=CONFIG_ENSEMBLE['device']
        )
    return inferencers


# ============================================================================
# TEMPLATE 10: Custom Configuration Builder
# ============================================================================

class InferencerBuilder:
    """Builder pattern for creating custom configurations."""
    
    def __init__(self):
        self.config = {
            'model_checkpoint': None,
            'data_folder': None,
            'device': 'cuda',
            'inference_params': {
                'top_k': 10,
                'threshold': 0.05,
            }
        }
    
    def set_checkpoint(self, path: str) -> 'InferencerBuilder':
        """Set model checkpoint path."""
        self.config['model_checkpoint'] = path
        return self
    
    def set_data_folder(self, path: str) -> 'InferencerBuilder':
        """Set data folder path."""
        self.config['data_folder'] = path
        return self
    
    def set_device(self, device: str) -> 'InferencerBuilder':
        """Set inference device (cuda or cpu)."""
        self.config['device'] = device
        return self
    
    def set_top_k(self, top_k: int) -> 'InferencerBuilder':
        """Set number of top candidates."""
        self.config['inference_params']['top_k'] = top_k
        return self
    
    def set_threshold(self, threshold: float) -> 'InferencerBuilder':
        """Set probability threshold."""
        self.config['inference_params']['threshold'] = threshold
        return self
    
    def build(self) -> HybridNSMInference:
        """Build and return configured inferencer."""
        return HybridNSMInference(
            model_checkpoint=self.config['model_checkpoint'],
            data_folder=self.config['data_folder'],
            device=self.config['device']
        )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("HYBRID NSM INFERENCE - CONFIGURATION TEMPLATES")
    print("=" * 80)
    
    # Example 1: Using predefined configurations
    print("\n[Example 1] Using predefined CWQ configuration")
    print("-" * 80)
    print(f"Config: {CONFIG_CWQ['description']}")
    print("Code:")
    print("""
    from hybrid_inference_config import setup_cwq_inferencer
    
    inferencer = setup_cwq_inferencer()
    inferencer.load_subgraph_batch('data/cwq/test_simple.json')
    results = inferencer.infer('0', 'Your query', top_k=10)
    """)
    
    # Example 2: Using builder pattern
    print("\n[Example 2] Using builder pattern for custom configuration")
    print("-" * 80)
    print("Code:")
    print("""
    from hybrid_inference_config import InferencerBuilder
    
    inferencer = (InferencerBuilder()
        .set_checkpoint('checkpoint/custom_model.ckpt')
        .set_data_folder('data/custom/')
        .set_device('cuda')
        .set_top_k(5)
        .set_threshold(0.1)
        .build())
    """)
    
    # Example 3: Selecting config based on environment
    print("\n[Example 3] Selecting configuration based on use case")
    print("-" * 80)
    print("Code:")
    print("""
    use_case = 'production'  # or 'development', 'cpu_only', etc.
    
    if use_case == 'production':
        inferencer = setup_production_inferencer()
    elif use_case == 'development':
        inferencer = setup_dev_inferencer()
    elif use_case == 'cpu_only':
        inferencer = setup_cpu_inferencer()
    """)
    
    # Example 4: Defining custom configuration
    print("\n[Example 4] Defining custom configuration")
    print("-" * 80)
    print("Code:")
    print("""
    MY_CONFIG = {
        'model_checkpoint': 'my_model.ckpt',
        'data_folder': 'my_data/',
        'device': 'cuda',
        'inference_params': {
            'top_k': 15,
            'threshold': 0.08,
        }
    }
    
    def setup_my_inferencer():
        return HybridNSMInference(
            model_checkpoint=MY_CONFIG['model_checkpoint'],
            data_folder=MY_CONFIG['data_folder'],
            device=MY_CONFIG['device']
        )
    """)
    
    print("\n" + "=" * 80)
    print("Available templates:")
    print("  1. setup_cwq_inferencer()           - CWQ Benchmark")
    print("  2. setup_webqsp_inferencer()        - WebQSP Benchmark")
    print("  3. setup_production_inferencer()    - Production (fast)")
    print("  4. setup_dev_inferencer()           - Development")
    print("  5. setup_cpu_inferencer()           - CPU only")
    print("  6. setup_batch_inferencer()         - Batch processing")
    print("  7. setup_kg_exploration_inferencer()- KG exploration")
    print("  8. setup_api_server_inferencer()    - API server")
    print("  9. setup_ensemble_inferencer()      - Multi-model ensemble")
    print(" 10. InferencerBuilder()              - Custom builder")
    print("=" * 80)
