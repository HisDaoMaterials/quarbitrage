"""
SQL Client API to connect, read, and write from and to databases.

CXDatabaseClient requires 'connectorx' library.
    - MYSQL write capabilities requires 'mysqlclient' library to interact with polars.
"""

import connectorx as cx
from typing import Union
from warnings import warn

from polars import DataFrame, LazyFrame

from quarbitrage.sql.database_client import DatabaseClient


class CXDatabaseClient(DatabaseClient):
    """
    Connectorx Database Client
    """

    def __init__(self, connection: Union[str, dict] = None):
        """ """
        super().__init__(connection=connection)

        self.set_connection_uri()
        self.create_connection_config()

    def set_connection_uri(self) -> None:
        """Set self.connection to uri if dictionary was provided"""
        if isinstance(self.connection_config, dict):
            self.connection = self.create_connection_uri(**self.connection_config)

    def create_connection_config(self) -> None:
        """
        Create self.connection_config if it hasn't been created
        """
        if self.connection_config is None:
            print(self.connection)
            try:
                split_uri = self.connection.split(":")
                password_host = split_uri[2].split("@")
                port_database = split_uri[-1].split("/")

                assert len(split_uri) == 4
                assert len(password_host) == 2
                assert 1 <= len(port_database) <= 2

                connection_config = {
                    "dbms": split_uri[0],
                    "user": split_uri[1].split("/")[-1],
                    "password": password_host[0],
                    "host": password_host[1],
                    "port": port_database[0],
                }

                if len(port_database) == 2:
                    connection_config["database"] = port_database[1]

                self.connection_config = connection_config
            except Exception as e:
                raise e(
                    "self.connection must adopt URI format 'dbms://user:password@host:port' or 'dbms://user:password@host:port/database'"
                )

    def read_query(
        self,
        query: str,
        lazy_evaluator: bool = True,
        protocol: str = "text",
        return_type: str = "polars",
        partition_num: int = 1,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Read query into DataFrame
        """
        df = cx.read_sql(
            conn=self.connection,
            query=query,
            protocol=protocol,
            return_type=return_type,
            partition_num=partition_num,
        )

        if (return_type in ["polars"]) and lazy_evaluator:
            return df.lazy()
        else:
            return df

    def read_table(
        self,
        table_name: str,
        database: str = None,
        lazy_evaluator: bool = True,
        return_type: str = "polars",
    ) -> Union[DataFrame, LazyFrame]:
        """Read table into DataFrame"""

        query = (
            f"SELECT * FROM {table_name}"
            if database is None
            else f"SELECT * FROM {database}.{table_name}"
        )

        return self.read_query(
            query, lazy_evaluator=lazy_evaluator, return_type=return_type
        )

    def write_table(
        self,
        df: Union[LazyFrame, DataFrame],
        table_name: str,
        database: str = None,
        insert_mode: str = "replace",
        create_database_if_not_exists: bool = True,
    ) -> None:
        """
        Write table onto database

        PARAMETERS
        ----------
            df: polars LazyFrame/DataFrame
                DataFrame to write contents onto relational database table.

            table_name: str
                Name of the table to create or append to in the target SQL database.

            database: str
                Name of database to write to.

            insert_mode: {'replace', 'append', 'fail'}
                The insert mode.
                    'replace': Create a new database table, overwriting an existing one.
                    'append': Append to an existing table.
                    'fail': Fail if table already exists.

            create_database_if_not_exists: bool
                Create database if it does not exist.
        """
        if self.connection_config.get("database", None) is None:
            if database is None:
                raise ValueError(
                    "The connection configuration doesn't specify a database. The 'database' argument is required, please specify it."
                )
            table_name = f"{database}.{table_name}"
        else:
            if database is not None:
                warn(
                    "Provided 'database' argument is not overriding database specified in connection configuration."
                )

        if create_database_if_not_exists:
            self.write_database(database=database)

        df.write_database(
            table_name=table_name,
            connection=self.connection,
            if_table_exists=insert_mode,
        )

    def write_database(self, database: str) -> None:
        """
        Create database on dbms if it does not exist.
        """
        if self.connection_config["dbms"] in ["mysql"]:
            import MySQLdb

            connection = MySQLdb.connect(
                host=self.connection_config["host"],
                user=self.connection_config["user"],
                password=self.connection_config["password"],
                port=self.connection_config["port"],
            )
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            connection.close()

    @staticmethod
    def create_connection_uri(dbms, user, password, host, port, database=None):
        """
        Create uri
        """
        uri = f"{dbms}://{user}:{password}@{host}:{port}"

        return uri if database is None else f"{uri}/{database}"
