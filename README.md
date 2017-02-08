This repository contains a set of tools to build a cluster 
based context-aware recommender system based on the data from UK
travel agency Helpful Holidays (HH).

How to prepare HH's data?
1. the folder `hh/cleaners` contains script to clean and convert HH's 
provided raw data to easy-to-process format
2. the script `hh/booking_split_and_transform.py` splits the booking
data into testing and training parts
3. the `feature_matrix` folder converts the easy-to-process format data
into feature-style format, i.e., presents each user/booking/item as a 
feature vector

The once prepared version of the data for users that have at least three
bookings can be found in the `data` folder.

To cluster the feature represented data use scripts from the `clustering`
folder.

To run the basic offline evaluation check the `experiment` folder.

The final recommender model can be found in the `model` folder 
(it works with all the available and requires separate data preparation).
