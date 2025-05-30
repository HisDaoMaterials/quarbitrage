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


def extract_file_format(file_path: str) -> str:
    """
    Extract file format of provided path
    """
    return file_path.split(".")[-1]


def get_filetype(file_path: str) -> str:
    """
    Get file type of provided file path
    """
    file_format = extract_file_format(file_path)
    
    if file_format in ["csv"]:
        file_type = "csv"
    elif file_format in ["xls", "xlsx", "xlsm", "xlsb"]:
        file_type = "excel"
    elif file_format in ["parquet"]:
        file_type = "parquet"
    elif file_format in ["json"]:
        file_type = "json"
    elif file_format in ["txt"]:
        file_type = "text"
    elif file_format in ["yaml"]:
        file_type = "yaml"
    else:
        file_type = None
    return file_type
