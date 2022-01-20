#!/usr/bin/env python

import sys
import argparse
import os.path
import SeqModules as sm
from tqdm import tqdm

def parse_args(argv = sys.argv[1:]):
	arg_parser = argparse.ArgumentParser()

	# optional arguments
	arg_parser.add_argument("-o", "--output",
		dest="output",
		metavar="STR",
		help="output file name; default stdout")
	arg_parser.add_argument("--disable-tqdm",
		dest="disable",
		action="store_true",
		help="disable tqdm progress bar")
	arg_parser.add_argument("-k",
		dest="k",
		metavar="INT",
		type=int,
		default=4,
		help="k for documenting; default 4")
	arg_parser.add_argument("-f", "--format",
		dest="format",
		choices=('fasta', 'fastq'),
		default='fasta',
		help="format of input file; default fasta")
	arg_parser.add_argument("-p", "--paired",
		dest="paired",
		metavar="FASTA/Q",
		help="paired fasta file; optional")
	arg_parser.add_argument("--min-length",
		dest="min_length",
		type=int,
		default=200,
		help="minimum allowed read length (sorter reads will be ignored)l default 200")
	arg_parser.add_argument("--max-length",
		dest="max_length",
		type=int,
		default=251,
		help="maximum allowed read length (longer reads will be trimmed); default 251")

	# required arguments
	req_group = arg_parser.add_argument_group("required arguments")
	req_group.add_argument("-i", "--input",
		dest="input",
		metavar="FASTA/Q",
		required=True,
		help="input fasta file; required")

	return arg_parser.parse_args(argv)

def seq2doc(seq, k, max_length):
	end_idx = min(len(seq), max_length) - k + 1
	return " ".join([seq[idx : idx + k] for idx in range(0, end_idx)])

def main(argv = sys.argv[1:]):
	args = parse_args(argv)

	ostream = open(args.output, 'w') if args.output else sys.stdout
	header = "forward,backward,seqid" if args.paired is not None else "sequence,seqid"
	print(header, file = ostream)

	parser = sm.SeqParser(args.input, fmt = args.format)
	num_seqs = parser.indexing(only_counts = True)
	fwds = parser.read()
	if args.paired is not None:
		paired_parser = sm.SeqParser(args.paired, fmt = args.format)
		bwds = paired_parser.read()
	
	if args.paired is not None:
		for fwd, bwd in tqdm(zip(fwds, bwds), total = num_seqs, disable = args.disable):
			sid1 = fwd['id']
			sid2 = bwd['id']
			if len(fwd['seq']) < args.min_length or \
				len(bwd['seq']) < args.min_length:
				continue

			doc1 = seq2doc(fwd['seq'], args.k, args.max_length)
			doc2 = seq2doc(bwd['seq'], args.k, args.max_length)

			doc = [doc1, doc2, fwd['id']]
			print(','.join(doc), file = ostream)
	else:
		for seq in tqdm(fwds, total = num_seqs, disable = args.disable):
			if len(seq['seq']) < args.min_length:
				continue

			doc = seq2doc(seq['seq'], args.k, args.max_length)
			doc = [doc, seq['id']]
			print(','.join(doc), file = ostream)

	if args.output:
		ostream.close()

	return 0

# main
if __name__ == "__main__":
	exit(main())
