"""
Configuration class for NSM model training and inference.
Extracted from run_CWQ.sh command arguments.
"""
import argparse


def merge_args_with_config(args: argparse.Namespace, config) -> argparse.Namespace:
    """
    Merge config class attributes with argparse.Namespace, updating the namespace with config values.
    
    Args:
        args: argparse.Namespace object to be updated
        config: Configuration class instance (e.g., NSMConfig or subclass)
    
    Returns:
        Updated argparse.Namespace object with config values merged in
    
    Example:
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=16)
        args = parser.parse_args(['--batch_size', '32'])
        
        config = CWQHybridTeacherConfig()
        config.batch_size = 20
        
        merged_args = merge_args_with_config(args, config)
        # merged_args.batch_size is now 20 (from config)
    """
    # Get config dict - handles both instance and class attributes
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        # Use vars() for instance attributes, then add class attributes
        config_dict = vars(config).copy() if hasattr(config, '__dict__') else {}
        # Add class-level attributes that aren't in __dict__
        for key in dir(config):
            if not key.startswith('_') and key not in config_dict:
                attr = getattr(config, key)
                # Only include non-callable attributes
                if not callable(attr):
                    config_dict[key] = attr
    
    for key, value in config_dict.items():
        # Merge all attributes from config, allowing later configs to override earlier ones
        setattr(args, key, value)
    
    return args


class NSMConfig:
    """Configuration class containing all arguments for NSM models."""
    
    def __init__(self):
        # Model configuration
        self.name = "CWQ"
        self.model_name = "gnn"
        self.teacher_model = "gnn"
        self.teacher_type = "hybrid"  # Options: hybrid, parallel
        
        # Data configuration
        self.data_folder = "/home/hegaole/data/KBQA/Freebase/CWQ/"
        self.word_emb_file = "word_emb_300d.npy"
        
        # Checkpoint configuration
        self.checkpoint_dir = "checkpoint/pretrain/"
        self.load_experiment = None
        self.load_pretrain = None
        self.load_teacher = None
        
        # Training configuration
        self.batch_size = 20
        self.test_batch_size = 40
        self.num_epoch = 100
        self.num_step = 4
        self.eval_every = 2
        self.lr = 5e-4
        self.eps = 0.95
        
        # Model dimensions
        self.entity_dim = 50
        self.word_dim = 300
        self.kg_dim = 100
        self.kge_dim = 100
        
        # Model type configurations
        self.q_type = "seq"
        self.loss_type = "kl"
        self.constrain_type = "js"
        
        # Regularization and loss weights
        self.lambda_constrain = 0.01
        self.lambda_back = 0.1
        self.lambda_label = 0.05
        
        # Flags (boolean arguments)
        self.use_self_loop = True
        self.reason_kb = True
        self.encode_type = True
        
        # Experiment name
        self.experiment_name = "CWQ_nsm"
    
    def __repr__(self):
        """String representation of the configuration."""
        config_items = []
        for key, value in self.__dict__.items():
            config_items.append(f"{key}={value}")
        return f"NSMConfig({', '.join(config_items)})"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()
    
    def update(self, **kwargs):
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"NSMConfig has no attribute '{key}'")
        return self
    
    @staticmethod
    def from_dict(config_dict):
        """Create configuration from dictionary."""
        config = NSMConfig()
        config.update(**config_dict)
        return config


# Preset configurations for different experiment types
class CWQHybridTeacherConfig:
    """Configuration for CWQ Hybrid Teacher model."""
    
    def __init__(self):
        self.checkpoint_dir = "checkpoint/CWQ_teacher/"
        self.load_experiment = "../pretrain/CWQ_nsm-final.ckpt"
        self.experiment_name = "CWQ_hybrid_teacher"
        self.num_epoch = 70
        self.teacher_type = "hybrid"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


class CWQHybridStudentConfig:
    """Configuration for CWQ Hybrid Student model."""
    
    def __init__(self):
        self.checkpoint_dir = "checkpoint/CWQ_student/"
        self.experiment_name = "CWQ_hybrid_student"
        self.num_epoch = 100
        self.teacher_type = "hybrid"
        self.load_teacher = "../CWQ_teacher/CWQ_hybrid_teacher-final.ckpt"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


class CWQParallelTeacherConfig:
    """Configuration for CWQ Parallel Teacher model."""
    
    def __init__(self):
        self.checkpoint_dir = "checkpoint/CWQ_teacher/"
        self.load_pretrain = "../pretrain/CWQ_nsm-final.ckpt"
        self.experiment_name = "CWQ_parallel_teacher"
        self.num_epoch = 30
        self.teacher_type = "parallel"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


class CWQParallelStudentConfig:
    """Configuration for CWQ Parallel Student model."""
    
    def __init__(self):
        self.checkpoint_dir = "checkpoint/CWQ_student/"
        self.experiment_name = "CWQ_parallel_student"
        self.num_epoch = 100
        self.teacher_type = "parallel"
        self.load_teacher = "../CWQ_teacher/CWQ_parallel_teacher-final.ckpt"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


# WebQSP Configurations
class WebQSPNSMConfig:
    """Configuration for WebQSP NSM model."""
    
    def __init__(self):
        self.name = "WebQSP"
        self.data_folder = "/home/hegaole/data/KBQA/Freebase/webqsp/"
        self.checkpoint_dir = "checkpoint/pretrain/"
        self.experiment_name = "webqsp_nsm"
        self.num_step = 3
        self.num_epoch = 200
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


class WebQSPHybridTeacherConfig:
    """Configuration for WebQSP Hybrid Teacher model."""
    
    def __init__(self):
        self.name = "WebQSP"
        self.data_folder = "/home/hegaole/data/KBQA/Freebase/webqsp/"
        self.checkpoint_dir = "checkpoint/webqsp_teacher/"
        self.load_experiment = "../pretrain/webqsp_nsm-final.ckpt"
        self.experiment_name = "webqsp_hybrid_teacher"
        self.num_step = 3
        self.num_epoch = 100
        self.teacher_type = "hybrid"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


class WebQSPHybridStudentConfig:
    """Configuration for WebQSP Hybrid Student model."""
    
    def __init__(self):
        self.name = "WebQSP"
        self.data_folder = "/home/hegaole/data/KBQA/Freebase/webqsp/"
        self.checkpoint_dir = "checkpoint/webqsp_student/"
        self.experiment_name = "webqsp_hybrid_student"
        self.num_step = 3
        self.num_epoch = 200
        self.teacher_type = "hybrid"
        self.load_teacher = "../webqsp_teacher/webqsp_hybrid_teacher-final.ckpt"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


class WebQSPParallelTeacherConfig:
    """Configuration for WebQSP Parallel Teacher model."""
    
    def __init__(self):
        self.name = "WebQSP"
        self.data_folder = "/home/hegaole/data/KBQA/Freebase/webqsp/"
        self.checkpoint_dir = "checkpoint/webqsp_teacher/"
        self.load_pretrain = "../pretrain/webqsp_nsm-final.ckpt"
        self.experiment_name = "webqsp_parallel_teacher"
        self.num_step = 3
        self.num_epoch = 100
        self.teacher_type = "parallel"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


class WebQSPParallelStudentConfig:
    """Configuration for WebQSP Parallel Student model."""
    
    def __init__(self):
        self.name = "WebQSP"
        self.data_folder = "/home/hegaole/data/KBQA/Freebase/webqsp/"
        self.checkpoint_dir = "checkpoint/webqsp_student/"
        self.experiment_name = "webqsp_parallel_student"
        self.num_step = 3
        self.num_epoch = 200
        self.teacher_type = "parallel"
        self.load_teacher = "../webqsp_teacher/webqsp_parallel_teacher-final.ckpt"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


if __name__ == "__main__":
    # Example usage
    config = CWQHybridTeacherConfig()
    print("CWQ Hybrid Teacher Configuration:")
    print(f"  checkpoint_dir={config.checkpoint_dir}")
    print(f"  load_experiment={config.load_experiment}")
    print(f"  num_epoch={config.num_epoch}")
    
    # Example of merging args with config
    print("\n\nExample of merging argparse.Namespace with config:")
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # Simulate command-line arguments
    args = parser.parse_args(['--batch_size', '32', '--lr', '1e-4'])
    print(f"Before merge: batch_size={args.batch_size}, num_epoch={args.num_epoch}, lr={args.lr}")
    
    # Merge with config
    hybrid_config = CWQHybridTeacherConfig()
    merged_args = merge_args_with_config(args, hybrid_config)
    print(f"After merge: batch_size={merged_args.batch_size}, num_epoch={merged_args.num_epoch}, lr={merged_args.lr}")


