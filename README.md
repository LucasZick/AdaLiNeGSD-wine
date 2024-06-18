To run, simply start any of the "train_adaline_sgd_m..." files. The difference between the two is as follows:

- __train_adaline_sgd_mono.py__ - plots a single combination of columns based on the user's choice (the columns are defined in the code in the column_x and column_y declarations). The generated image is not stored.

- __train_adaline_sgd_multiple.py__ - plots all possible combinations in the database avoiding repetitions, stores them all individually within the "plots" folder and, finally, generates the file "_all_plots_.pdf", containing all the images together for a clear and convenient analysis.