
import pandas as pd

def convert_to_datetime(series):
    return pd.to_datetime(series, errors='coerce')
