import logging

import numpy as np
import pandas as pd
from datetime import datetime


class TimeWindowSplitter(object):
    def __init__(self, window=1.0, split_point=0.2):
        """The objects of this class split the array of timestamps into two
         parts (e.g., training and testing) using time-based criteria.

        :param window: percentage of a data's time window that should be split. By default 1.0 (100%)
        :param split_point: splitting border. By default 0.2.
            It means that the first 80% of the time window we go to the first part, while
            the following 20% to second part.
        """
        self.window = window
        self.split_point = split_point

    def split(self, timestamps, random_state=None):
        """

        :param timestamps: list, array or pandas.Series object containing timestamps
        :param random_state: random state for windows selection
        :return: indexes of timestamps related to the first and the second parts
        """
        if random_state is not None:
            np.random.seed(random_state)

        logging.info(u"Starting to split data")
        dt_col = pd.Series(timestamps)

        min_dt_ord = dt_col.min().toordinal()
        max_dt_ord = dt_col.max().toordinal()

        delta = int((max_dt_ord - min_dt_ord) * self.window)
        start_dt_ord = np.random.randint(min_dt_ord, max(max_dt_ord - delta, min_dt_ord + 1))
        end_dt_ord = start_dt_ord + delta

        start_dt = datetime.fromordinal(start_dt_ord)
        end_dt = datetime.fromordinal(end_dt_ord)
        border_dt = datetime.fromordinal(end_dt_ord - int(delta * self.split_point))
        logging.info(u"Timestamps: start=%s, end=%s, border=%s", start_dt, end_dt, border_dt)

        part1 = np.where((dt_col >= start_dt) & (dt_col < border_dt))[0]
        part2 = np.where((dt_col >= border_dt) & (dt_col <= end_dt))[0]

        logging.info(u"Data has been split")
        return part1, part2
