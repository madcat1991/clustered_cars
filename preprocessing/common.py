import logging

import pandas as pd


def canonize_datetime(df, columns):
    """Canonizes datetime fields
    """
    logging.info(u"Converting to datetime: %s", columns)
    df[columns] = df[columns].apply(lambda x: pd.to_datetime(x, dayfirst=True, infer_datetime_format=True))
    return df


def canonize_float(df, columns):
    """Canonizes float fields
    """
    logging.info(u"Converting to float: %s", columns)
    for col in columns:
        logging.debug(u"Canonizing float in %s", col)
        df[col] = df[col].apply(
            lambda x: float(x.replace(",", ".") if isinstance(x, str) else x)
        )
    return df
