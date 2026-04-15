import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging

logger = logging.getLogger(__name__)


# =========================
# MC DROPOUT LAYER
# =========================
class MCDropout(nn.Dropout):
    """MC Dropout always active during inference."""

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True, inplace=self.inplace)


# =========================
# RESNET MODEL (CLEAN API)
# =========================
class ResNetWithMCDropout(nn.Module):

    def __init__(self,
                 backbone='resnet18',
                 num_classes=4,
                 embedding_dim=512,
                 dropout_rate=0.3,
                 pretrained=True):
        super().__init__()

        # Backbone
        if backbone == 'resnet18':
            net = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            net = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = nn.Sequential(*list(net.children())[:-1])

        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            MCDropout(p=dropout_rate)
        )

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.embedding_dim = embedding_dim

    # =========================
    # MAIN FORWARD
    # ALWAYS RETURNS LOGITS ONLY
    # =========================
    def forward(self, x):
        emb = self.extract_embeddings(x)
        logits = self.classifier(emb)
        return logits

    # =========================
    # FOR TRAINER (OPTIONAL)
    # =========================
    def forward_with_embedding(self, x):
        emb = self.extract_embeddings(x)
        logits = self.classifier(emb)
        return logits, emb

    # =========================
    # EMBEDDING EXTRACTOR
    # =========================
    def extract_embeddings(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        return emb


# =========================
# DENSENET MODEL (SAME API)
# =========================
class DenseNetWithMCDropout(nn.Module):

    def __init__(self,
                 num_classes=4,
                 embedding_dim=512,
                 dropout_rate=0.3,
                 pretrained=True):
        super().__init__()

        net = models.densenet121(pretrained=pretrained)
        self.backbone = net.features

        feature_dim = 1024

        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            MCDropout(p=dropout_rate)
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.embedding_dim = embedding_dim

    def forward(self, x):
        emb = self.extract_embeddings(x)
        return self.classifier(emb)

    def forward_with_embedding(self, x):
        emb = self.extract_embeddings(x)
        logits = self.classifier(emb)
        return logits, emb

    def extract_embeddings(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.embedding(x)


# =========================
# FACTORY FUNCTION
# =========================
def create_model(config: dict, device: torch.device):

    backbone = config['model']['backbone']
    num_classes = config['model']['num_classes']
    embedding_dim = config['model']['embedding_dim']
    dropout_rate = config['model']['dropout_rate']
    pretrained = config['model']['pretrained']

    if backbone in ['resnet18', 'resnet50']:

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
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    return model