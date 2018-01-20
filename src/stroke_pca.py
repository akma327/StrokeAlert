# StrokeAlert

import sys
import numpy as np
from sklearn.decomposition import PCA

stroke_data = "/Users/anthony/Desktop/hackathon/StrokeAlert/data/IST_corrected.csv"
variable_labels = "/Users/anthony/Desktop/hackathon/StrokeAlert/data/IST_variables.csv"

def process_data():
	"""
	Parse the complete stroke trial database and retain relevant columns. 
	Data cleaning phase
	"""
	print "process_data()"
	

def stroke_pca():
	"""
	Perform a pca on the the filtered dataset. 
	"""
	X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	pca = PCA(n_components=2)
	pca.fit(X)
	print(pca.explained_variance_ratio_) 

if __name__ == "__main__":
	stroke_pca()