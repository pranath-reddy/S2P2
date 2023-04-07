'''
Name: Pranath Reddy Kumbam
UFID: 8512-0977
NLP Project Codebase

Code for saving the UC Berkley "Measuring Hate Speech" dataset from Hugginface for manual data exploration
'''

# Load libraries
import datasets 

# Load the dataset
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
df = dataset['train'].to_pandas()
df.describe()

# Save the dataset
df.to_csv('./HFdata.csv')
