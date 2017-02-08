This repository contains a set of tools to build a cluster
based context-aware recommender system using the data provided by UK
travel agency Helpful Holidays (HH).

How to prepare HH's data?
1. the folder `hh/cleaners` contains scripts to clean and convert HH's
raw data into an easy-to-process format
2. the script `hh/booking_split_and_transform.py` splits the booking
data into testing and training parts
3. the `feature_matrix` folder contains scripts that convert
the easy-to-process data into a feature-style format, i.e.,
presents each user/booking/item as a feature vector

A once prepared version of the data for the users that have at
least three bookings can be found in the `data` folder.

To cluster the feature represented data use scripts from the `clustering`
folder.

To run the basic offline evaluation check the `experiment` folder.

The final recommender model can be found in the `model` folder
(it works with all the available dat and requires separate
data preparation process).
