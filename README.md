### Introduction

This repository contains a set of tools to build a cluster based
context-aware recommender system the UK based travel agency Helpful
Holidays (HH).

### Data preprocessing

HH provides us with datasets full of raw data. So, before creating the
model, data should be preprocessed to machine-friendly format.

How to do this?

1. the folder `hh/cleaners` contains scripts to clean and convert HH's
data sets into an easy-to-process format
2. the script `hh/booking_split_and_transform.py` splits the prepared
bookings into testing and training parts
3. the `feature_matrix` folder contains scripts that convert
the easy-to-process data into a feature-style format, i.e.,
presents each user/booking as a feature vector

### Clustering

The scripts from the `clustering` folder can be used to cluster
users/bookings represented as binary feature matrices.

### Evaluation

To run the basic offline evaluation check the `experiment` folder.

### Example

An example recommender model can be found in the `model` folder.
