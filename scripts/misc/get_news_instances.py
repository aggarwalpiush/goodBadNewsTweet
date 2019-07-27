#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import sys

def load_file(infile):
	in_data = []
	with codecs.open(infile, 'r', 'utf-8') as in_obj:
		for line in in_obj:
			line = line.rstrip('\r\n').split('\t')
			in_data.append(line)
	return in_data


def main():
	pred_prob_file = sys.argv[1]
	ref_file = sys.argv[2]
	threshold_value = float(sys.argv[3])

	pred_data = load_file(pred_prob_file)
	ref_data = load_file(ref_file)

	assert len(pred_data) == len(ref_data)


	with codecs.open(ref_file+'_f8_q.tsv', 'w', 'utf-8') as out_obj:
		for i, val in enumerate(pred_data):
			if float(val[1]) >= threshold_value:
				out_obj.write('\t'.join(ref_data[i])+'\n')





if __name__ == '__main__':
	main()
