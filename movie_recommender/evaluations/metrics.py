from typing import List

__all__ = ['evaluate_recall']


def evaluate_recall(pred_items: List[int], gt_items: List[int]) -> dict:
    """calculate recall on pred items and gt items

    Args:
        pred_items: pred items
        gt_items: gt items

    Returns:
        dict: metric
    """
    gt_items = set(gt_items)
    return {
        'metric_type':
        'recall',
        'metric_value':
        len(gt_items.intersection(pred_items)) /
        len(gt_items) if pred_items else None
    }
