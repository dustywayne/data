import pandas as pd
import numpy as np
from IPython import embed as shell

# read independent variables
df = pd.read_csv('banking/independent_variables.csv')

age = np.array(df['age'], dtype=str)
job = np.array(df['job'], dtype=str)
marital = np.array(df['marital'], dtype=str)
education = np.array(df['education'], dtype=str)

loan = np.array(pd.read_csv('banking/dependent_variable.csv')['loan'], dtype=str)
loan = [a == 'yes' for a in loan]
