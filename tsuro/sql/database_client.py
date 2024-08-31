from abc import ABC, abstractmethod
from typing import Union
from polars import DataFrame, LazyFrame


class DatabaseClient(ABC):
    """
    Abstract base class that contains the structure for connecting to SQL databases, reading &
    writing tables/databases.
    """

    def __init__(self, connection_config: dict = None):
        """
        Constructor for DatabaseClient
        """
        self.connection_config: dict = connection_config
        self.connection = None

    def set_connection_config(self, connection_config: dict) -> None:
        """Set configuration for database connection"""
        self.connection_config = connection_config
        self.connect()

    @abstractmethod
    def connect(self) -> None:
        """Create connection / cursor object to database"""

    @abstractmethod
    def read_query(self, query: str, lazy_evaluator: bool = True) -> Union[LazyFrame, DataFrame]:
        """Read SQL query into Polars LazyFrame / DataFrame"""

    @abstractmethod
    def read_table(self, lazy_evaluator: bool = True) -> Union[LazyFrame, DataFrame]:
        """Read table from database"""

    @abstractmethod
    def write_table(self, pdf: Union[LazyFrame, DataFrame]) -> None:
        """Write dataframe to table on database"""

    @abstractmethod
    def write_database(self) -> None:
        """Create database on database"""
