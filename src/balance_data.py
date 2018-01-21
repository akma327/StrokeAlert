# StrokeAlert
import sys

USAGE_STR = """

python balance_data.py /Users/anthony/Desktop/hackathon/StrokeAlert/data/cause_of_death/full_data_set.csv /Users/anthony/Desktop/hackathon/StrokeAlert/data/cause_of_death/balanced_data_set.csv 

"""

def balance_data(input_file, output_file):
	f = open(input_file, 'r')
	fo = open(output_file, 'w')
	header = f.readline()
	alive_count, dead_count = 0, 0
	max_count = 5000
	for line in f:
		linfo = line.strip().split(",")
		life_state = int(linfo[-1])
		if(life_state == 1): 
			if(alive_count < max_count):
				fo.write(line)
				alive_count +=1
		else:
			if(dead_count < max_count):
				fo.write(line)
				dead_count +=1
	


if __name__ == "__main__":
	(input_file, output_file) = (sys.argv[1], sys.argv[2])
	balance_data(input_file, output_file)