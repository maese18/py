# This example shows how data cleaning can be done with pandas library
# Source: https://towardsdatascience.com/a-complete-machine-learning-walk-through-in-python-part-one-c62152f39420, 2018-12-23
# Github: https://github.com/WillKoehrsen/machine-learning-project-walkthrough

import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# Read in data into a dataframe
data = pd.read_csv('data/20180605-871m-data.csv')
# Display top of dataframe
data.head()