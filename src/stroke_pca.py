# StrokeAlert

import sys
import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


stroke_data = "../data/IST_cleaned.csv"
variable_labels = "../data/IST_variables.csv"
normalized_stroke_data = "../data/IST_normalized.csv"

def process_data():
	"""
	Parse the complete stroke trial database and retain relevant columns. 
	Data cleaning phase
	"""
	f = open(stroke_data, 'r')
	header = f.readline().split(",")
	data = []
	for line in f:
		data.append(map(float, line.split(",")))
	return header, np.array(data)


def stroke_pca():
	"""
	Perform a pca on the the filtered dataset. 
	"""
	header, data = process_data()

	data_std = StandardScaler().fit_transform(data)
	f = open(normalized_stroke_data, 'w')
	f.write(",".join(header))
	for line in data_std:
		f.write(",".join(map(str,line)) + "\n")


	print data_std, data_std.shape, data.shape
	# pca = PCA(n_components=30)
	# pca.fit_transform(data)
	# print(pca.components_)
	# print(pca.explained_variance_ratio_) 

if __name__ == "__main__":
	stroke_pca()