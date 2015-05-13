## Data Mining K-Means Algorithm

This program implements the K-Means clustering algorithm.
This program is designed to run with a 'data.txt' file within a '/.data' directory.
If different filenames are to be used the code will need to be updated accordingly.


**Build & Execute Instructions**

To build and execute run command `python kmeans.py` from the command line.

You will be presented with 4 options:

1. Perform K-Means clustering using cluster means as cluster centre.
2. Perform K-Means clustering using closest insatnce to mean as cluster centre.
3. Automate option 1 for K = 2->20 with 5 iterations on each value of K.
4. Automate option 2 for K = 2->20 with 5 iterations on each value of K.

Note: _Options 3 & 4 just allow the automated repetition of options 1 or 2.
They carry out K-Means clustering with values of K ranging from 2 to 20.
For each value of K, the process is repeated 5 times and average precision, recall and f_score calculated.
These scores are then printed at the end. This is useful for generating graphs such as in the report._
