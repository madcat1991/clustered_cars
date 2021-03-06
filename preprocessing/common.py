import logging
from collections import Counter

import pandas as pd


def canonize_datetime(df, columns):
    """Canonizes datetime fields
    """
    logging.info(u"Converting to datetime: %s", columns)
    df[columns] = df[columns].apply(lambda x: pd.to_datetime(x, dayfirst=True, infer_datetime_format=True))
    return df


def raw_data_to_df(path, delimiter):
    """Canonizing the process of reading raw data from files

    :param path: path to raw data file
    :param delimiter: columns separator
    :return: pd.DataFrame object
    """
    df = pd.read_csv(path, sep=delimiter)
    df.columns = map(str.lower, df.columns)
    return df


def check_processed_columns(processed, original):
    """Checks whether all original columns were processed"""
    _processed = set(processed)
    _original = set(original)

    if len(_processed) > len(_original):
        diff = _processed.difference(_original)
        logging.error("The number processed columns is higher than original. "
                      "Strange processed columns: %s", diff)
        raise Exception("Wrong processed columns")
    elif len(_processed) < len(_original):
        diff = _original.difference(_processed)
        logging.error("Several original columns have not been processed: %s", diff)
        raise Exception("Columns have not been processed")
    else:
        diff = _original.difference(_processed)
        if diff:
            logging.error("Several original columns have not been processed: %s", diff)
            raise Exception("Columns have not been processed")


def check_data(path, sep):
    number_of_cols = []

    with open(path) as f:
        for line in f:
            number_of_cols.append(len(line.split(sep)))

    rows_per_number = Counter(number_of_cols)

    if len(rows_per_number) != 1:
        logging.error("Inconsistent number of columns: %s", rows_per_number)

        outlier = sorted(rows_per_number.items(), key=lambda x: x[1])[0][0]
        with open(path) as f:
            for line in f:
                if len(line.split(sep)) == outlier:
                    logging.info("Outlier=%s, example:\n%s", outlier, line)
                    break
        exit(1)
    logging.info("Data check complete: SUCCESS")
