import torch
import yaml
import logging

from train import ContinualLearningTrainer
from src.data_loader import create_dataloaders


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("CL")


if __name__ == "__main__":

    config = load_config()
    logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = ContinualLearningTrainer(config, device, logger)

    train_tasks, val_tasks = create_dataloaders(config)

    for task_id, task_data in enumerate(train_tasks):

        train_loader, val_loader, classes = task_data

        logger.info(f"\n===== STARTING TASK {task_id} =====")

        acc = trainer.train_task(
            task_id=task_id,
            train_loader=train_loader,
            val_loader=val_loader,
            task_classes=classes
        )

        logger.info(f"Task {task_id} completed | Best Acc: {acc:.4f}")

    logger.info("ALL TASKS COMPLETED")