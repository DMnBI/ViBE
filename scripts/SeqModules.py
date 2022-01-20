__author__  = "hjgwak"
__version__ = "1.0.1"
__date__    = "2020. 09. 15"

import pickle as pkl
import pandas as pd
import numpy as np

try:
	from tqdm.auto import tqdm
	tqdm_available = True
except:
	tqdm_available = False

class SeqParser:
	"""
	sequence file parser
	"""

	# constructor and destructor
	def __init__ (self, file_name = None, fmt = "fastq", encoding = 'utf-8'):
		self.__file_stream = open(file_name, 'r', encoding = encoding) if file_name else None
		self.__format = fmt
		self.__index = {}

	def __del__ (self):
		self.close()

	# file stream manager
	def open (self, file_name, fmt = "fastq", encoding = 'utf-8'):
		self.close()
		self.__file_stream = open(file_name, 'r', encoding = encoding)
		self.__format = fmt

	def close (self):
		if self.is_open():
			self.__index = {}
			self.__file_stream.close()

	def is_open (self):
		return (self.__file_stream and not self.__file_stream.closed)

	def file_name (self):
		if self.is_open():
			return self.__file_stream.name
		# below return statement will be replaced by exception handling
		return None

	# getters
	def format (self):
		return self.__format

	def read (self):
		self.__file_stream.seek(0)
		remain = True
		while remain:
			record = self.__read_one_seq_from_file()
			if record == None:
				remain = False
				break

			yield record

	# exception handling is needed
	def get (self, sid):
		if sid in self.__index:
			self.__file_stream.seek(self.__index[sid])
			record = self.__read_one_seq_from_file()
			return record

		return None

	#TODO exception handling is needed
	def indexing (self, disable_tqdm = True, only_counts = False):
		def get_num_seqs(file_stream, fmt):
			file_stream.seek(0)
			total = 0
			if fmt == "fasta":
				total = len([1 for line in tqdm(file_stream.readlines(), disable = disable_tqdm, desc = "counting reads") if line.startswith('>')])
			elif fmt == 'fastq':
				total = len([1 for line in tqdm(file_stream.readlines(), disable = disable_tqdm, desc = "counting reads")]) / 4

			return int(total)

		def get_sid(line):
			return line.rstrip("\r\n").split(" ")[0].split('\t')[0][1:]

		def parse_fasta(stream, index, pbar = None):
			prev = stream.tell()
			while True:
				line = stream.readline()
				if line == "":
					break

				if line.startswith('>'):
					index[get_sid(line)] = prev
					if pbar:
						pbar.update(1)
				prev = stream.tell()

		def parse_fastq(stream, index, pbar = None):
			prev = stream.tell()
			while True:
				header = stream.readline()
				if header == "":
					break

				index[get_sid(header)] = prev
				for i in range(3):
					_ = stream.readline()
				prev = stream.tell()
				if pbar:
					pbar.update(1)

		if not self.is_open():
			exit(-1)

		num_seqs = get_num_seqs(self.__file_stream, self.__format)
		if only_counts:
			return num_seqs

		self.__file_stream.seek(0)
		self.__index = dict()
		pbar = tqdm(total = num_seqs, desc = f"Indexing {self.file_name()}", disable = disable_tqdm) if tqdm_available else None
		if self.__format == "fasta":
			parse_fasta(self.__file_stream, self.__index, pbar = pbar)
		elif self.__format == "fastq":
			parse_fastq(self.__file_stream, self.__index, pbar = pbar)
		if pbar:
			pbar.close()

		self.__file_stream.seek(0)

	def index_dump (self, file_name):
		with open(file_name, 'wb') as o:
			pkl.dump(self.__index, o)

	def index_load (self, file_name):
		with open(file_name, 'rb') as f:
			self.__index = pkl.load(f)
			
	def get_sid_list(self):
		if self.__index == {}:
			self.indexing()

		return list(self.__index.keys())

	# In this version, parser assumes taking correctly formatted file only
	def __read_one_seq_from_file (self):
		if self.__format == "fasta":
			header = self.__file_stream.readline()
			if header == "":
				return None

			header = header.rstrip('\r\n')[1:]
			sid = header.split(' ')[0].split('\t')[0]
			seq = ""

			while True:
				prev = self.__file_stream.tell()
				tmp = self.__file_stream.readline()
				if tmp.startswith(">"):
					self.__file_stream.seek(prev)
					break
				if tmp == "":
					break
				seq += tmp.rstrip('\r\n')
			record = {"header" : header, "id" : sid, "seq" : seq}

			return record
		elif self.__format == "fastq":
			header = self.__file_stream.readline().rstrip("\r\n")[1:]
			sid = header.split(' ')[0]
			seq = self.__file_stream.readline().rstrip("\r\n")
			desc = self.__file_stream.readline().rstrip("\r\n")
			qual = self.__file_stream.readline().rstrip("\r\n")

			if header == "" or seq == "" or desc == "" or qual == "":
				return None
			
			record = {"header" : header, "id" : sid, "seq" : seq, "desc" : desc, "quality" : qual}

			return record

		return None

# simple functions
def format(record, fmt):
	tmp = ""
	if fmt == "fasta":
		tmp += ">" + record["header"] + "\n"
		tmp += record["seq"]
	elif fmt == "fastq":
		tmp += "@" + record["header"] + "\n"
		tmp += record["seq"] + "\n"
		tmp += record["desc"] + "\n"
		tmp += record["quality"]

	return tmp

