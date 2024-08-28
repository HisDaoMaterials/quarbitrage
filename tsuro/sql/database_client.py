from abc import ABC, abstractmethod
from typing import Union
from polars import DataFrame, LazyFrame


class DatabaseClient(ABC):
    """
    Abstract base class that contains the structure for connecting to SQL databases, reading & writing tables/databases.
    """

    def __init__(self):
        """
        Constructor for DatabaseClient
        """
        self.connection_config: dict = None
        self.connection = None

    def set_connection_config(self, connection_config: dict) -> None:
        """Set configuration for database connection"""
        self.connection_config = connection_config

    @abstractmethod
    def connect(self) -> None:
        """Create connection / cursor object to database"""

    @abstractmethod
    def read_table(self) -> Union[DataFrame, LazyFrame]:
        """Read table from database"""

    @abstractmethod
    def write_table(self, pdf: Union[DataFrame, LazyFrame], table_name: str) -> None:
        """Write dataframe to table on database"""

    @abstractmethod
    def write_database(self) -> None:
        """Create database on database"""
