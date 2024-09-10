def fetch_table(self, *args, **kwargs) -> Union[LazyFrame, DataFrame]:
    """
    Read table into Polars LazyFrame / DataFrame using DatabaseClient.read_table()

    Parameters:
        *args: Positional arguments relevant for self.database_client.read_table()
        **kwargs: Keyword arguments relevant for self.database_client.read_table()
    """
    if self.database_client is None:
        print(
            "No .database_client attribute provided. Please provide DatabaseClient via .set_database_client() method"
        )
    else:
        self.pdf = self.database_client.read_table(
            *args, **kwargs, lazy_evaluator=self.lazy_evaluator
        )

def fetch_query(self, *args, **kwargs) -> Union[LazyFrame, DataFrame]:
    """
    Read SQL query into polars LazyFrame or DataFrame using DatabaseClient.read_query()

    Parameters:
        *args: Positional arguments for self.database_client.read_query()
        **kwargs: Keyword arguments for self.database_client.read_query()
    """
    if self.database_client is None:
        print(
            "No .database_client attribute provided. Please provide DatabaseClient via .set_database_client() method"
        )
    else:
        self.pdf = self.database_client.read_query(
            *args, **kwargs, lazy_evaluator=self.lazy_evaluator
        )