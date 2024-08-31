"""
Module for reading, extracting, formatting .csv, .excel, .parquet, and .txt files
"""


def extract_csv_row_to_list(
    csv_row: str, to_upper: bool = True, delimiter: str = ",", trim: bool = False
) -> list[str]:
    """
    Extract single csv row into python list of entries
    """
    csv_row = csv_row.split("\n")[0]
    if to_upper:
        csv_row = csv_row.upper()

    csv_row_list = csv_row.split(delimiter)

    if trim:
        csv_row_list = [entry.strip() for entry in csv_row_list]

    return csv_row_list
