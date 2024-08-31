"""
ImbalanceBars Class
"""

from tsuro.sql import DatabaseClient
from tsuro.data_structures.bars import Bars


class ImbalanceBars(Bars):
    """
    ImbalanceBars Class
    """
    
    def __init__(
        self,
        database_client: DatabaseClient = None,
        datetime_col: str = "DATETIME",
        price_col: str = "PRICE",
        volume_col: str = "VOLUME",
    ):
        """
        Constructor for Imbalance Bars
        """
        super().__init__(
            database_client=database_client,
            datetime_col=datetime_col,
            price_col=price_col,
            volume_col=volume_col,
        )
