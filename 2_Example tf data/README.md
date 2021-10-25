Example using tf data to train a model with data that might not fit on RAM,
by reading from multiple files and shuffling.

- create_folders_of_data.py: Create the data to use in the example


- tf_data_statswithdask.py: Use dask to calculate the mean and standard deviation of the data, to use to normalize it


- tf_data.py: Use tf.data to read, interleave, and shuffle data. Then train in a CNN with regularization