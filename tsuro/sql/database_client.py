from abc import ABC, abstractmethod
from typing import Union
from polars import DataFrame, LazyFrame
from tsuro.utils.file_handling import extract_file_format


class DatabaseClient(ABC):
    """
    Abstract base class that contains the structure for connecting to SQL databases, and reading and
    writing tables/databases.
    """

    def __init__(self, connection: Union[str, dict] = None):
        """
        Constructor for DatabaseClient.

        Parameters
        ----------
            connection_config: str/dict
                If a string, it should denote the database uri, or the file path to a yaml file.
                If a dictionary, it should contain the relevant keywords to establish a connection to a database.
        """
        self.connection: Union[str, dict] = connection
        self.connection_config: dict = None

        self.set_connection_config()

    def set_connection_config(self) -> None:
        """Set configuration for database connection"""

        if isinstance(self.connection, str):
            if extract_file_format(self.connection) in ["yaml"]:
                self._set_config_from_yaml(self.connection)
        elif isinstance(self.connection, dict):
            self.connection_config = self.connection

    def _set_config_from_yaml(self, filepath_yaml: str) -> None:
        """Set connection configuration from yaml file"""
        import yaml

        with open(filepath_yaml, "r") as stream:
            self.connection_config = yaml.safe_load(stream)

    @abstractmethod
    def read_query(self, query: str, *args, **kwargs) -> Union[LazyFrame, DataFrame]:
        """Read SQL query into Polars LazyFrame / DataFrame"""

    @abstractmethod
    def read_table(self, *args, **kwargs) -> Union[LazyFrame, DataFrame]:
        """Read table from database"""

    @abstractmethod
    def write_table(self, df: Union[LazyFrame, DataFrame], *args, **kwargs) -> None:
        """Write dataframe to table on database"""

    @abstractmethod
    def write_database(self, *args, **kwargs) -> None:
        """Create new database workspace on your database"""
