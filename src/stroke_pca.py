# StrokeAlert

import sys
import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

stroke_data = "../data/IST_cleaned.csv"
variable_labels = "../data/IST_variables.csv"

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
	X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	pca = PCA(n_components=30)
	pca.fit_transform(data)
	print(pca.components_)
	# print(pca.explained_variance_ratio_) 

if __name__ == "__main__":
	stroke_pca()