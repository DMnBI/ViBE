#!/usr/bin/env python

import sys
import argparse
import os.path
import pandas as pd

def parse_args(argv = sys.argv[1:]):
	arg_parser = argparse.ArgumentParser()

	# optional arguments
	arg_parser.add_argument("-o", "--output-dir",
		dest="o_dir",
		metavar="PATH",
		default=".",
		help="directory to save outputs; default .")
	arg_parser.add_argument("-c", "--cutoff",
		dest="cutoff",
		metavar="FLOAT",
		type=float,
		help="confidence score cutoff")
	arg_parser.add_argument("-t", "--targets",
		dest="targets",
		metavar="LABEL",
		nargs="*",
		help="a list of labels of target; if this option is specified, only specified labels will be reported")

	# required arguments
	req_group = arg_parser.add_argument_group("required arguments")
	req_group.add_argument("-i", "--query",
		dest="query",
		metavar="CSV",
		required=True,
		help="query csv file; required")
	req_group.add_argument("-p", "--prediction",
		dest="preds",
		metavar="TXT",
		required=True,
		help="ViBE classification result file; required")

	return arg_parser.parse_args(argv)

def main(argv = sys.argv[1:]):
	args = parse_args(argv)

	preds = pd.read_csv(args.preds, sep = "\t")
	records = pd.read_csv(args.query, sep = ",")

	records = pd.merge(records, preds, how = 'inner', on = 'seqid')
	if args.targets is not None:
		records = records.loc[records['prediction'].isin(args.targets)]
	if args.cutoff is not None:
		records = records.loc[records['score'] >= args.cutoff]

	labels = set(records['prediction'])
	for label in labels:
		sub = records.loc[records['prediction'] == label]
		sub = sub.drop(['prediction', 'score'], axis = 1)
		sub.to_csv(f"{args.o_dir}/{label}.csv", sep = ",", index = False)

	return 0

# main
if __name__ == "__main__":
	exit(main())
