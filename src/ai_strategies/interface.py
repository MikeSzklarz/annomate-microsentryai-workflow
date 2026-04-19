"""
Abstract Strategy Interface for anomaly detection models.

Defines the contract all concrete strategy implementations must fulfill,
ensuring a consistent API for model loading and inference.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AnomalyDetectionStrategy(ABC):
    """Abstract base class defining the contract for anomaly detection strategies.

    All concrete strategy implementations must inherit from this class and
    implement :meth:`load_from_folder` and :meth:`predict`. The interface
    is intentionally kept minimal to support hot-swappable backends.

    Attributes:
        model_name (str): Human-readable identifier for the loaded model.
            Concrete subclasses should overwrite this after loading.
    """

    def __init__(self) -> None:
        """Initialize the strategy with a placeholder model name."""
        self.model_name = "Unknown"

    @abstractmethod
    def load_from_folder(self, folder_path: str) -> None:
        """Load model weights from *folder_path*.

        Args:
            folder_path (str): Path to the directory containing model weights
                and any required configuration files.

        Raises:
            FileNotFoundError: If *folder_path* or required weight files do
                not exist.
            RuntimeError: If the model cannot be initialised from the provided
                files.
        """

    @abstractmethod
    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """Run inference on a single image and return an anomaly score and heatmap.

        Args:
            image_path (str): Absolute path to the input image file.

        Returns:
            Tuple[float, np.ndarray]: A ``(anomaly_score, heatmap)`` pair
                where *anomaly_score* is a scalar float and *heatmap* is a
                2-D ``float32`` NumPy array with values normalised to 0–1.
        """
