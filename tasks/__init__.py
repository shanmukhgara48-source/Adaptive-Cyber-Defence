from .base import BaseTask, TaskConfig, TaskResult
from .easy import EasyTask
from .medium import MediumTask
from .hard import HardTask
from .nightmare import NightmareTask
from .elite import EliteTask
from .impossible import ImpossibleTask

__all__ = [
    "BaseTask", "TaskConfig", "TaskResult",
    "EasyTask", "MediumTask", "HardTask", "NightmareTask",
    "EliteTask", "ImpossibleTask",
]
