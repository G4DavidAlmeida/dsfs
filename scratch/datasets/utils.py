from typing import Union
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum

class BaseDatasetsNames(Enum):
    def val_name(self):
        if isinstance(self.value, str):
            return self.value
        elif isinstance(self.value, tuple):
            return self.value[0]
        else:
            raise ValueError('Unknown value type')

class BaseDatasets(ABC):
    @abstractmethod
    def _download(self, dataset_name: BaseDatasetsNames):
        pass

class BaseDirPaths:
    def __init__(self, base_dir: Union[Path, str]):
        self.base_dir = Path(base_dir)
        if not self.base_dir.is_dir():
            raise NotADirectoryError(f'{base_dir} is not a directory')
        
    def _check_file(self, file_path: Path):
        if not file_path.is_file():
            raise FileNotFoundError(f'{file_path} not found')
        return file_path