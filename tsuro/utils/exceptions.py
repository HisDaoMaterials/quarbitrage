class UnequalLengthError(ValueError):
    """Warning class used to notify the user that the two lists/dictionaries don't have equal length"""

    def __init__(self, length1: int, length2: int, alias1: str, alias2: str):
        self.length1 = length1
        self.length2 = length2
        self.alias1 = alias1
        self.alias2 = alias2

    def __str__(self):
        return f"Length('{self.alias1}')={self.length1} != Length('{self.alias2}')={self.length2}. Please ensure same length for both objects."
