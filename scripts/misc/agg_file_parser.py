#! usr/bin python
# -*- coding : utf-8 -*-

import codecs
import sys

def main():
	input_file = sys.argv[1]
	output_file = input_file+'.out'

	with codecs.open(output_file,'w','utf-8') as out_obj:
		with codecs.open(input_file, 'r', 'utf-8') as in_obj:
			outline = []
			keywords = []
			i = -1
			for line in in_obj:
				tokens = line.split(',')
				if len(tokens) >= 5:
					if '"' in tokens[4].strip('\r\n'):
						outline.append('	'.join(tokens[:4]))
						if tokens[4].strip('\r\n''"') != '':
							keywords.append(tokens[4].strip('\r\n''"'))
						i += 1
						continue
				elif '"' not in line and len(tokens) < 5:
					if line.strip('\r\n') == '':
						continue
					keywords.append(line.strip('\r\n'))
					continue
				elif len(tokens) < 5 and '"' in line:
					keywords.append(line.strip('\r\n''"'))
					outline[i] = outline[i] + '\t' + ','.join(keywords)
					keywords = []
					out_obj.write(outline[i]+'\n')
					continue


if __name__ == '__main__':
	main()



