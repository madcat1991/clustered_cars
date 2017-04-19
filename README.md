# HH recommender system

This repository contains a set of tools to build a cluster based
context-aware recommender system the UK based travel agency Helpful
Holidays (HH).

## Data preprocessing

HH provides us with datasets full of raw data. So, before creating the
model, data should be preprocessed to machine-friendly format.

The folder `preprocessing` contains scripts to clean and convert HH's
data sets into an easy-to-process format. The `feature_matrix` folder
contains scripts that convert the easy-to-process data into a
feature-style format, i.e., presents each user/booking as a feature
vector.

## Transforming bookings

Before converting the cleaned bookings to the booking-feature matrix
it should be transformed. Since the transformation depends on the amount
of data two transformation scripts presented. The first script is
`evaluation/booking_transform_and_split.py` and it is used for splitting
data into the training and testing datasets.

The second script is used to build the final model and can be found in
the `model` folder.

## Feature space

To convert transformed bookings and cleaned properties/contacts/features
into the feature representation the scripts from the `feature_matrix`
sholud be used.

## Clustering

The scripts from the `clustering` folder can be used to cluster
users/bookings represented as binary feature matrices.

### Faiss

We use `faiss` library from Facebook to efficiently cluster dense
vectors. For the work of the library three files should be presented in
the root directory: `faiss.py`, `swigfaiss.py` and `_swigfaiss.so`.
Check [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
for more details.

## Evaluation

To run the basic offline evaluation check the `evaluation` folder.

The script `booking_transform_and_split.py` transforms the cleaned
booking data and splits it into the testing and training parts. To
convert resulting training booking into the feature representation use
the `feature_matrix/booking.py` script setting parameter `-b` to the csv
file with the training bookings.

## Example

An example recommender model can be found in the `model` folder.
