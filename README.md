# ML_Basics

# Machine Learning Basics with Python

Welcome to this beginner-friendly guide to machine learning using Python! This README will cover the basics of popular data science libraries such as NumPy, pandas, Matplotlib, and Seaborn. We'll also walk you through how to collect datasets, import data from Kaggle, and handle missing values.

## Table of Contents
1. [Introduction to NumPy](#introduction-to-numpy)
2. [Introduction to pandas](#introduction-to-pandas)
3. [Introduction to Matplotlib](#introduction-to-matplotlib)
4. [Introduction to Seaborn](#introduction-to-seaborn)
5. [Collecting Datasets](#collecting-datasets)
6. [Importing Data from Kaggle](#importing-data-from-kaggle)


## Introduction to NumPy

NumPy is a powerful library for numerical computations in Python. It provides support for arrays, matrices, and many mathematical functions.


# Performing basic operations
print("Sum:", np.sum(array))
print("Mean:", np.mean(array))
print("Standard Deviation:", np.std(array))

## Introduction to Pandas

pandas is a library used for data manipulation and analysis. It provides data structures like DataFrame that are easy to use for handling structured data.
import pandas as pd

# Creating a DataFrame
data = {'Name': ['John', 'Anna', 'Peter'], 'Age': [28, 24, 35]}
df = pd.DataFrame(data)
print(df)

# Basic DataFrame operations
print("DataFrame shape:", df.shape)
print("First few rows:\n", df.head())
print("Descriptive statistics:\n", df.describe())

## Introduction to Matplotlib
Matplotlib is a plotting library used for creating static, animated, and interactive visualizations in Python.

import matplotlib.pyplot as plt

# Creating a simple plot
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Plot')
plt.show()

[You can view the notebook directly here](https://colab.research.google.com/github/Rashimanish/ML_Basics/blob/main/Matplotlib.ipynb)

##Introduction to Seaborn
Seaborn is a statistical data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

import seaborn as sns

# Load a sample dataset
tips = sns.load_dataset('tips')

# Create a scatter plot
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.title('Total Bill vs Tip')
plt.show()

## Importing Data from Kaggle
To import data from Kaggle, you need to use the Kaggle API. Here are the steps:

# install kaggle 
!pip install kaggle

# upload Kaggle API Key:

Go to your Kaggle account, create a new API token, and download the kaggle.json file.
Upload the kaggle.json file to your project directory.

# Configure Kaggle API:
import os
# Create a .kaggle directory
!mkdir -p ~/.kaggle
# Move kaggle.json to .kaggle directory
!cp kaggle.json ~/.kaggle/
# Set permissions for the kaggle.json file
!chmod 600 ~/.kaggle/kaggle.json

#download data set
!kaggle datasets download -d <dataset-identifier>

#unzip data set
!unzip <dataset-file>.zip
