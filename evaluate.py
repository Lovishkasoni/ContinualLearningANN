import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import logging
from pathlib import Path
import json

from src import get_device, load_config


logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation and visualization for continual learning."""
    
    def __init__(self, config: dict, metrics, device: torch.device):
        self.config = config
        self.metrics = metrics
        self.device = device
        self.plot_dir = Path(config['logging']['plot_dir'])
        self.metrics_dir = Path(config['logging']['metrics_dir'])
        
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_task_accuracy_heatmap(self):
        """Plot task-wise accuracy heatmap."""
        logger.info("Plotting task accuracy heatmap...")
        
        accuracy_matrix = self.metrics.get_task_accuracy_matrix()
        
        if accuracy_matrix is None or accuracy_matrix.size == 0:
            logger.warning("No accuracy data to plot")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=range(accuracy_matrix.shape[1]),
                   yticklabels=range(accuracy_matrix.shape[0]),
                   cbar_kws={'label': 'Accuracy'})
        plt.xlabel('Task Learned')
        plt.ylabel('Task Evaluated')
        plt.title('Task-wise Accuracy Heatmap (Forward & Backward Transfer)')
        plt.tight_layout()
        
        save_path = self.plot_dir / 'accuracy_heatmap.png'
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved to {save_path}")
        plt.close()
    
    def plot_accuracy_per_task(self):
        """Plot accuracy curves per task."""
        logger.info("Plotting accuracy per task...")
        
        if not self.metrics.task_accuracies:
            logger.warning("No task accuracies to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        for task_id, epochs_acc in self.metrics.task_accuracies.items():
            if epochs_acc:
                epochs = sorted(epochs_acc.keys())
                accuracies = [epochs_acc[e] for e in epochs]
                plt.plot(epochs, accuracies, marker='o', label=f'Task {task_id}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Per Task Over Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.plot_dir / 'accuracy_per_task.png'
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved to {save_path}")
        plt.close()
    
    def plot_drift_timeline(self, drift_events: list):
        """Plot drift detection timeline."""
        logger.info("Plotting drift detection timeline...")
        
        if not drift_events:
            logger.info("No drift events to plot")
            return
        
        drift_batches = [e['batch_idx'] for e in drift_events]
        drift_means = [e['mean'] for e in drift_events]
        drift_stds = [e['std'] for e in drift_events]
        
        plt.figure(figsize=(12, 5))
        plt.scatter(drift_batches, drift_means, s=100, alpha=0.6, color='red', label='Drift Events')
        plt.errorbar(drift_batches, drift_means, yerr=drift_stds, fmt='none', 
                    ecolor='red', alpha=0.3, capsize=5)
        
        plt.xlabel('Batch Index')
        plt.ylabel('Embedding Mean')
        plt.title('Drift Detection Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.plot_dir / 'drift_timeline.png'
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved to {save_path}")
        plt.close()
    
    def plot_buffer_utilization(self, buffer_stats_history: list):
        """Plot buffer utilization over time."""
        logger.info("Plotting buffer utilization...")
        
        if not buffer_stats_history:
            logger.info("No buffer stats to plot")
            return
        
        utilizations = [s['utilization'] for s in buffer_stats_history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(utilizations, marker='o', linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Buffer Utilization')
        plt.title('Privacy-Preserving Replay Buffer Utilization')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.plot_dir / 'buffer_utilization.png'
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, num_tasks: int):
        """Plot confusion matrices for each task."""
        logger.info("Plotting confusion matrices...")
        
        num_cols = 3
        num_rows = (num_tasks + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
        if num_tasks == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for task_id in range(num_tasks):
            if task_id in self.metrics.confusion_matrices:
                cm = self.metrics.confusion_matrices[task_id]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[task_id],
                           cbar_kws={'label': 'Count'})
                axes[task_id].set_title(f'Task {task_id}')
                axes[task_id].set_ylabel('True Label')
                axes[task_id].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(num_tasks, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        save_path = self.plot_dir / 'confusion_matrices.png'
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved to {save_path}")
        plt.close()
    
    def save_metrics_summary(self, results: dict):
        """Save metrics summary to JSON."""
        logger.info("Saving metrics summary...")
        
        summary = self.metrics.get_summary()
        summary.update(results)
        
        save_path = self.metrics_dir / 'metrics_summary.json'
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved to {save_path}")
    
    def generate_results_table(self, results: dict) -> str:
        """Generate results summary table."""
        logger.info("Generating results table...")
        
        table_str = "\n" + "="*80 + "\n"
        table_str += "CONTINUAL LEARNING RESULTS SUMMARY\n"
        table_str += "="*80 + "\n\n"
        
        # Task accuracies
        table_str += "Per-Task Accuracies:\n"
        table_str += "-" * 40 + "\n"
        for task_id, metrics_dict in results.items():
            table_str += f"Task {task_id}:\n"
            table_str += f"  Val Accuracy:  {metrics_dict.get('val_accuracy', 0):.4f}\n"
            table_str += f"  Test Accuracy: {metrics_dict.get('test_accuracy', 0):.4f}\n"
        
        # Forgetting
        table_str += "\n" + "-" * 40 + "\n"
        table_str += "Forgetting per Task:\n"
        table_str += "-" * 40 + "\n"
        for task_id in self.metrics.task_accuracies:
            forgetting = self.metrics.compute_forgetting(task_id)
            table_str += f"Task {task_id}: {forgetting:.4f}\n"
        
        # Transfer
        table_str += "\n" + "-" * 40 + "\n"
        table_str += "Transfer Metrics:\n"
        table_str += "-" * 40 + "\n"
        table_str += f"Average Forward Transfer: {self.metrics.get_summary()['avg_forward_transfer']:.4f}\n"
        table_str += f"Average Backward Transfer: {self.metrics.get_summary()['avg_backward_transfer']:.4f}\n"
        
        table_str += "\n" + "="*80 + "\n"
        
        return table_str
