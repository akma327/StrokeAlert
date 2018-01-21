# StrokeAlert
import sys

USAGE_STR = """

python balance_data.py /Users/anthony/Desktop/hackathon/StrokeAlert/data/cause_of_death/full_data_set.csv /Users/anthony/Desktop/hackathon/StrokeAlert/data/cause_of_death/balanced_data_set.csv 

"""

def balance_data(input_file, output_file):
	f = open(input_file, 'r')
	header = f.readline()
	alive_count, dead_count = 0, 0
	for line in f:
		linfo = line.strip().split(",")
		print linfo

if __name__ == "__main__":
	(input_file, output_file) = (sys.argv[1], sys.argv[2])
	balance_data(input_file, output_file)