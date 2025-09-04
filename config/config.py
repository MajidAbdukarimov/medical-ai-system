# config/config.py
import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    """Data configuration"""
    train_csv: str
    val_csv: str
    test_csv: str
    img_dir: str
    batch_size: int = 32
    num_workers: int = 4
    
@dataclass
class ModelConfig:
    """Model configuration"""
    backbone: str = 'resnet50'
    num_classes: int = 14
    pretrained: bool = True
    use_attention: bool = True
    dropout_rate: float = 0.5
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    lr: float = None
    
    def __post_init__(self):
        """Post initialization to handle lr alias"""
        if self.lr is None:
            self.lr = self.learning_rate
        else:
            self.learning_rate = self.lr
    
@dataclass
class Config:
    """Main configuration"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment_name: str
    seed: int = 42
    
def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    # Open with UTF-8 encoding to avoid issues
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(
        data=DataConfig(**config_dict['data']),
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        experiment_name=config_dict['experiment_name'],
        seed=config_dict.get('seed', 42)
    )