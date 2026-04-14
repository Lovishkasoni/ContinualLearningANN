import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging


logger = logging.getLogger(__name__)


class MCDropout(nn.Dropout):
    """MC Dropout layer - applies dropout at inference time."""
    
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True, inplace=self.inplace)


class ResNetWithMCDropout(nn.Module):
    """ResNet with MC Dropout for uncertainty estimation."""
    
    def __init__(self, backbone: str = 'resnet18', num_classes: int = 4,
                 embedding_dim: int = 512, dropout_rate: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Embedding layer with MC Dropout
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            MCDropout(p=dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x, return_embedding: bool = False):
        """Forward pass."""
        # Backbone
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Embedding
        embeddings = self.embedding(features)
        
        # Classification
        logits = self.classifier(embeddings)
        
        if return_embedding:
            return logits, embeddings
        return logits
    
    def extract_embeddings(self, x):
        """Extract embeddings without classification."""
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.embedding(features)
        return embeddings
    
    def get_penultimate_layer(self):
        """Get penultimate layer for regularization."""
        return self.embedding[-2]  # Linear layer before dropout


class DenseNetWithMCDropout(nn.Module):
    """DenseNet with MC Dropout for uncertainty estimation."""
    
    def __init__(self, num_classes: int = 4, embedding_dim: int = 512,
                 dropout_rate: float = 0.3, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = models.densenet121(pretrained=pretrained)
        feature_dim = 1024
        
        # Embedding layer with MC Dropout
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            MCDropout(p=dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x, return_embedding: bool = False):
        """Forward pass."""
        # Backbone
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Embedding
        embeddings = self.embedding(features)
        
        # Classification
        logits = self.classifier(embeddings)
        
        if return_embedding:
            return logits, embeddings
        return logits
    
    def extract_embeddings(self, x):
        """Extract embeddings without classification."""
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        embeddings = self.embedding(features)
        return embeddings
    
    def get_penultimate_layer(self):
        """Get penultimate layer for regularization."""
        return self.embedding[-2]  # Linear layer before dropout


def create_model(config: dict, device: torch.device):
    """Create model based on config."""
    backbone = config['model']['backbone']
    num_classes = config['model']['num_classes']
    embedding_dim = config['model']['embedding_dim']
    dropout_rate = config['model']['dropout_rate']
    pretrained = config['model']['pretrained']
    
    if backbone == 'resnet18' or backbone == 'resnet50':
        model = ResNetWithMCDropout(
            backbone=backbone,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    elif backbone == 'densenet121':
        model = DenseNetWithMCDropout(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    model = model.to(device)
    
    logger.info(f"Model: {backbone}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model
