import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import logging


logger = logging.getLogger(__name__)


class ContinualLearningMetrics:
    """Metrics for evaluating continual learning."""
    
    def __init__(self):
        self.task_accuracies = {}  # {task_id: {epoch: accuracy}}
        self.forward_transfer = {}  # Forward transfer per task
        self.backward_transfer = {}  # Backward transfer per task
        self.confusion_matrices = {}  # {task_id: confusion_matrix}
        self.predictions_history = {}  # Store predictions for analysis
    
    def update_task_accuracy(self, task_id: int, epoch: int, accuracy: float):
        """Update accuracy for a task."""
        if task_id not in self.task_accuracies:
            self.task_accuracies[task_id] = {}
        self.task_accuracies[task_id][epoch] = accuracy
    
    def compute_forward_transfer(self, task_id: int, current_accuracy: float,
                               initial_accuracy: float) -> float:
        """
        Forward Transfer: F_t = A_t^t - A_t^0
        Performance on new task compared to initial random guess.
        """
        ft = current_accuracy - initial_accuracy
        self.forward_transfer[task_id] = ft
        return ft
    
    def compute_backward_transfer(self, task_id: int, 
                                 current_accuracies: dict,
                                 previous_accuracies: dict) -> float:
        """
        Backward Transfer: B_t = A_i^t - A_i^t-1
        Change in performance on old tasks after learning new task.
        """
        bt = 0.0
        count = 0
        
        for old_task_id in previous_accuracies:
            if old_task_id in current_accuracies:
                bt += current_accuracies[old_task_id] - previous_accuracies[old_task_id]
                count += 1
        
        if count > 0:
            bt = bt / count
        
        self.backward_transfer[task_id] = bt
        return bt
    
    def compute_average_accuracy(self, accuracies: dict) -> float:
        """Compute average accuracy across all tasks."""
        if not accuracies:
            return 0.0
        return np.mean(list(accuracies.values()))
    
    def update_confusion_matrix(self, task_id: int, predictions: np.ndarray,
                               targets: np.ndarray, num_classes: int):
        """Update confusion matrix for a task."""
        cm = confusion_matrix(targets, predictions, labels=np.arange(num_classes))
        self.confusion_matrices[task_id] = cm
    
    def store_predictions(self, task_id: int, predictions: np.ndarray,
                         targets: np.ndarray, probabilities: np.ndarray = None):
        """Store predictions for later analysis."""
        if task_id not in self.predictions_history:
            self.predictions_history[task_id] = {
                'predictions': [],
                'targets': [],
                'probabilities': []
            }
        
        self.predictions_history[task_id]['predictions'].append(predictions)
        self.predictions_history[task_id]['targets'].append(targets)
        if probabilities is not None:
            self.predictions_history[task_id]['probabilities'].append(probabilities)
    
    def get_task_accuracy_matrix(self) -> np.ndarray:
        """
        Get matrix where (i,j) = accuracy on task i after training on task j.
        This shows forward and backward transfer.
        """
        if not self.task_accuracies:
            return None
        
        num_tasks = len(self.task_accuracies)
        matrix = np.zeros((num_tasks, num_tasks))
        
        # This would be populated during training
        # For now, return what we have
        for task_id, epochs in self.task_accuracies.items():
            if epochs:
                final_accuracy = list(epochs.values())[-1]
                matrix[task_id, task_id] = final_accuracy
        
        return matrix
    
    def compute_forgetting(self, task_id: int) -> float:
        """
        Compute forgetting on task_id after learning subsequent tasks.
        """
        if task_id not in self.task_accuracies:
            return 0.0
        
        epochs = sorted(self.task_accuracies[task_id].keys())
        if len(epochs) < 2:
            return 0.0
        
        initial = self.task_accuracies[task_id][epochs[0]]
        final = self.task_accuracies[task_id][epochs[-1]]
        
        return initial - final
    
    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        summary = {
            'task_accuracies': self.task_accuracies,
            'forward_transfer': self.forward_transfer,
            'backward_transfer': self.backward_transfer,
            'avg_forward_transfer': np.mean(list(self.forward_transfer.values())) 
                                   if self.forward_transfer else 0.0,
            'avg_backward_transfer': np.mean(list(self.backward_transfer.values()))
                                   if self.backward_transfer else 0.0,
        }
        return summary


def compute_mc_dropout_uncertainty(logits_list: list) -> torch.Tensor:
    """
    Compute MC Dropout uncertainty from multiple forward passes.
    
    Args:
        logits_list: List of logit tensors from different MC samples
    
    Returns:
        Uncertainty (predictive variance)
    """
    # Stack logits: [num_samples, batch_size, num_classes]
    logits_stacked = torch.stack(logits_list, dim=0)
    
    # Compute softmax probabilities
    probs = torch.softmax(logits_stacked, dim=-1)
    
    # Compute mean and variance
    mean_probs = probs.mean(dim=0)  # [batch_size, num_classes]
    variance = torch.var(probs, dim=0)  # [batch_size, num_classes]
    
    # Aggregate uncertainty (mean across classes)
    uncertainty = variance.mean(dim=1)  # [batch_size]
    
    return uncertainty
