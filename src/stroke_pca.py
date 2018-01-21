# StrokeAlert

import sys
import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

USAGE_STR = """

python stroke_pca.py /Users/anthony/Desktop/hackathon/StrokeAlert/data/cause_of_death/x_features.csv /Users/anthony/Desktop/hackathon/StrokeAlert/data/cause_of_death/x_features_normalized.csv
"""


stroke_data = "../data/IST_cleaned.csv"
variable_labels = "../data/IST_variables.csv"
normalized_stroke_data = "../data/IST_normalized.csv"

def process_data(input_file):
	"""
	Parse the complete stroke trial database and retain relevant columns. 
	Data cleaning phase
	"""
	f = open(input_file, 'r')
	header = f.readline().split(",")
	data = []
	for line in f:
		data.append(map(float, line.split(",")))
	return header, np.array(data)

def normalize_data(input_file, output_file):
	header, data = process_data(input_file)

	data_std = StandardScaler().fit_transform(data)
	f = open(output_file, 'w')
	f.write(",".join(header))
	for line in data_std:
		f.write(",".join(map(str,line)) + "\n")

	return data_std

def stroke_pca():
	"""
	Perform a pca on the the filtered dataset. 
	"""
	header, data = process_data(stroke_data)

	data_std = normalize_data(input_file, output_file)
	
	# pca = PCA(n_components=30)
	# pca.fit_transform(data)
	# print(pca.components_)
	# print(pca.explained_variance_ratio_) 

if __name__ == "__main__":
	(input_file, output_file) = (sys.argv[1], sys.argv[2])
	normalize_data(input_file, output_file)
	# stroke_pca()

