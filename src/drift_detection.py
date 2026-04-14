import torch
import numpy as np
import logging
from collections import deque
from typing import List, Tuple


logger = logging.getLogger(__name__)


class PageHinkleyDetector:
    """Page-Hinkley drift detector for embedding streams."""
    
    def __init__(self, threshold: float = 3.0, window_size: int = 50,
                 min_instances: int = 30):
        """
        Args:
            threshold: Threshold for drift detection
            window_size: Size of moving window
            min_instances: Minimum instances before detecting drift
        """
        self.threshold = threshold
        self.window_size = window_size
        self.min_instances = min_instances
        
        self.running_mean = None
        self.running_variance = None
        self.mh_values = deque(maxlen=window_size)
        self.instance_count = 0
        self.drift_detected = False
        self.drift_count = 0
    
    def update(self, value: float) -> Tuple[bool, float]:
        """
        Update detector with new value and check for drift.
        
        Args:
            value: New value (usually embedding mean or variance)
        
        Returns:
            (drift_detected, statistic)
        """
        self.instance_count += 1
        
        # Initialize
        if self.running_mean is None:
            self.running_mean = value
            self.running_variance = 0.0
        
        # Update running statistics
        delta = value - self.running_mean
        self.running_mean = self.running_mean + delta / self.instance_count
        
        if self.instance_count > 1:
            delta2 = value - self.running_mean
            self.running_variance = (self.running_variance * (self.instance_count - 2) +
                                    delta * delta2) / (self.instance_count - 1)
        
        # Compute Page-Hinkley statistic
        std_dev = np.sqrt(max(self.running_variance, 1e-10))
        mh = (value - self.running_mean) / std_dev if std_dev > 0 else 0
        
        self.mh_values.append(mh)
        
        # Check drift condition
        drift = False
        if (self.instance_count > self.min_instances and 
            len(self.mh_values) == self.window_size):
            
            mh_sum = sum(self.mh_values)
            
            if abs(mh_sum) > self.threshold:
                drift = True
                self.drift_count += 1
                logger.warning(f"Drift detected! (count: {self.drift_count}, statistic: {mh_sum:.3f})")
        
        return drift, mh if len(self.mh_values) == self.window_size else 0.0
    
    def reset(self):
        """Reset detector."""
        self.running_mean = None
        self.running_variance = None
        self.mh_values.clear()
        self.instance_count = 0
        self.drift_detected = False
    
    def get_statistics(self) -> dict:
        """Get detector statistics."""
        return {
            'instance_count': self.instance_count,
            'drift_count': self.drift_count,
            'running_mean': float(self.running_mean) if self.running_mean is not None else None,
            'running_variance': float(self.running_variance) if self.running_variance is not None else None,
        }


class DriftDetector:
    """Main drift detector class."""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.drift_config = config['drift_detection']
        self.device = device
        
        self.detector = PageHinkleyDetector(
            threshold=self.drift_config['drift_threshold'],
            window_size=self.drift_config['window_size']
        )
        
        self.drift_events = []
        self.embedding_history = []
    
    def compute_embedding_statistics(self, embeddings: torch.Tensor) -> dict:
        """Compute statistics on embeddings."""
        embeddings = embeddings.detach().cpu().numpy()
        
        stats = {
            'mean': np.mean(embeddings),
            'std': np.std(embeddings),
            'median': np.median(embeddings),
            'q95': np.percentile(embeddings, 95),
            'range': np.max(embeddings) - np.min(embeddings),
        }
        
        return stats
    
    def detect_drift(self, embeddings: torch.Tensor, batch_idx: int = None) -> Tuple[bool, dict]:
        """
        Detect drift in embeddings.
        
        Args:
            embeddings: Batch of embeddings [batch_size, embedding_dim]
            batch_idx: Batch index for logging
        
        Returns:
            (drift_detected, stats)
        """
        stats = self.compute_embedding_statistics(embeddings)
        
        # Use mean as test statistic
        test_stat = stats['mean']
        
        drift_detected, ph_stat = self.detector.update(test_stat)
        
        stats['ph_statistic'] = float(ph_stat)
        stats['drift_detected'] = drift_detected
        stats['batch_idx'] = batch_idx
        
        if drift_detected:
            self.drift_events.append({
                'batch_idx': batch_idx,
                'mean': test_stat,
                'std': stats['std'],
                'timestamp': len(self.drift_events)
            })
            logger.info(f"Drift event #{len(self.drift_events)} at batch {batch_idx}")
        
        self.embedding_history.append(stats)
        
        return drift_detected, stats
    
    def reset_for_task(self):
        """Reset detector for new task."""
        self.detector.reset()
    
    def get_drift_timeline(self):
        """Get timeline of drift events."""
        return self.drift_events
    
    def get_statistics(self):
        """Get detector statistics."""
        return {
            'total_drifts': len(self.drift_events),
            'detector_stats': self.detector.get_statistics(),
            'embedding_history_length': len(self.embedding_history),
        }
