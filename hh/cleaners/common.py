# coding: utf-8
import logging

import pandas as pd


def drop_null(df, columns):
    """Drops null values from not null cols
    """
    logging.info(u"Omitting rows with NA in: %s", columns)
    return df.dropna(subset=columns)


def canonize_datetime(df, columns):
    """Canonizes datetime fields
    """
    logging.info(u"Converting to datetime: %s", columns)
    for col in columns:
        logging.debug(u"Canonizing datetime in %s", col)
        df[col] = pd.to_datetime(df[col], dayfirst=True)
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
