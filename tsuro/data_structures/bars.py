from abc import ABC, abstractmethod
from typing import Union
from polars import DataFrame, LazyFrame

class Bars(ABC):

    def __init__(self):
        """
        """
        self.connection = None
    
    @abstractmethod
    def create_bars(self) -> None:
        