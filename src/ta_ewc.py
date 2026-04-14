import torch
import torch.nn as nn
import numpy as np
import logging


logger = logging.getLogger(__name__)


class TaskAwareEWC:
    """Task-Aware Elastic Weight Consolidation with clinical importance weighting."""
    
    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.ewc_config = config['ewc']
        
        self.fisher_matrix = {}
        self.optimal_params = {}
        self.previous_fisher = {}
        self.parameter_importance_groups = {}
        
        self.task_id = 0
    
    def compute_fisher_information_matrix(self, data_loader, task_id: int = None):
        """Compute Fisher Information Matrix on current task data."""
        logger.info("Computing Fisher Information Matrix...")
        
        if task_id is None:
            task_id = self.task_id
        
        self.model.eval()
        
        fisher_matrix = {name: torch.zeros_like(param).to(self.device)
                        for name, param in self.model.named_parameters()
                        if param.requires_grad}
        
        sample_count = 0
        max_samples = self.ewc_config['fim_sample_size']
        
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            if sample_count >= max_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.model.zero_grad()
            
            # Forward pass
            logits, _ = self.model(images, return_embedding=True)
            
            # Compute log predictive distribution
            log_probs = torch.log_softmax(logits, dim=1)
            
            # For each sample
            for i in range(images.size(0)):
                self.model.zero_grad()
                
                sample_log_prob = log_probs[i, labels[i]]
                
                # Backward pass for this sample
                sample_log_prob.backward(retain_graph=True)
                
                # Accumulate squared gradients (Fisher Information)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_matrix[name] += param.grad.data ** 2
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
        
        # Normalize by number of samples
        for name in fisher_matrix:
            fisher_matrix[name] /= max_samples
            fisher_matrix[name] += self.ewc_config['fisher_eps']  # Add small constant
        
        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
        
        # Classify parameters into importance groups
        if self.ewc_config['task_aware']:
            self._classify_parameter_importance(fisher_matrix, task_id)
        
        # Store fisher matrix
        self.previous_fisher = self.fisher_matrix.copy()
        self.fisher_matrix = fisher_matrix
        self.task_id = task_id + 1
        
        logger.info(f"Fisher Information Matrix computed for task {task_id}")
        logger.info(f"Parameter importance groups: {self._summarize_importance_groups()}")
    
    def _classify_parameter_importance(self, fisher_matrix: dict, task_id: int):
        """Classify parameters into importance groups based on Fisher values and variance."""
        logger.info("Classifying parameters into importance groups...")
        
        self.parameter_importance_groups = {
            'clinically_critical': [],    # CC: High FIM + High variance
            'shared_representational': [], # SR: Moderate FIM + Low variance
            'task_peripheral': []         # TP: Low FIM
        }
        
        all_fisher_values = []
        fisher_by_layer = {}
        
        # Collect all fisher values and group by layer
        for name, fisher in fisher_matrix.items():
            layer_name = name.split('.')[0]
            fisher_mean = fisher.mean().item()
            all_fisher_values.append(fisher_mean)
            
            if layer_name not in fisher_by_layer:
                fisher_by_layer[layer_name] = []
            fisher_by_layer[layer_name].append(fisher_mean)
        
        # Compute thresholds
        all_fisher_values = np.array(all_fisher_values)
        high_threshold = np.percentile(all_fisher_values, 75)
        low_threshold = np.percentile(all_fisher_values, 25)
        
        # Classify parameters
        for name, fisher in fisher_matrix.items():
            fisher_mean = fisher.mean().item()
            
            # Compute variance across tasks (if previous fisher exists)
            if self.previous_fisher and name in self.previous_fisher:
                fisher_variance = torch.var(
                    torch.stack([self.previous_fisher[name], fisher])
                ).item()
            else:
                fisher_variance = 0.0
            
            # Classify
            if fisher_mean > high_threshold:
                if fisher_variance > np.percentile([f.mean().item() for f in self.previous_fisher.values()] 
                                                  if self.previous_fisher else [0], 50):
                    self.parameter_importance_groups['clinically_critical'].append(name)
                else:
                    self.parameter_importance_groups['shared_representational'].append(name)
            elif fisher_mean > low_threshold:
                self.parameter_importance_groups['shared_representational'].append(name)
            else:
                self.parameter_importance_groups['task_peripheral'].append(name)
        
        logger.info(f"  Clinically Critical: {len(self.parameter_importance_groups['clinically_critical'])} params")
        logger.info(f"  Shared Representational: {len(self.parameter_importance_groups['shared_representational'])} params")
        logger.info(f"  Task Peripheral: {len(self.parameter_importance_groups['task_peripheral'])} params")
    
    def _summarize_importance_groups(self) -> str:
        """Summarize importance groups."""
        return (f"CC:{len(self.parameter_importance_groups.get('clinically_critical', []))} "
                f"SR:{len(self.parameter_importance_groups.get('shared_representational', []))} "
                f"TP:{len(self.parameter_importance_groups.get('task_peripheral', []))}")
    
    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.fisher_matrix:
            return torch.tensor(0.0, device=self.device)
        
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in model.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                optimal_param = self.optimal_params[name]
                
                # Determine lambda based on importance group
                if self.ewc_config['task_aware']:
                    if name in self.parameter_importance_groups.get('clinically_critical', []):
                        lambda_param = self.ewc_config['lambda_cc']
                    elif name in self.parameter_importance_groups.get('shared_representational', []):
                        lambda_param = self.ewc_config['lambda_sr']
                    else:
                        lambda_param = self.ewc_config['lambda_tp']
                else:
                    lambda_param = self.ewc_config['lambda_cc']
                
                # EWC loss: λ * Σ F_i * (θ_i - θ*_i)²
                ewc_loss += (lambda_param / 2) * torch.sum(
                    fisher * (param - optimal_param) ** 2
                )
        
        return ewc_loss
    
    def reset(self):
        """Reset EWC for new task."""
        # Don't reset fisher_matrix - we keep history for variance computation
        pass
