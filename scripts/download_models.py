import sys
import argparse
import subprocess as sp

from tqdm import tqdm

FILEID_MAP = {
	"pre-trained": {"FILEID": "100EITt7ZmyjkBl_X1kJ83nfV5jpK_ED1",
					"MD5SUM": "10f2fb1a433f87f7508834e4e8a32733"},
	"BPDR150": {"FILEID": "1nSTwkvfeJ5VTs2__FOIVW9IO-L8iQZid",
				"MD5SUM": "13e46abc7cd1d6987fc0279e3f39c034"},
	"BPDR250": {"FILEID": "1WdawuAiz1E4CYwrtjvd24dNFHUjns9ZZ",
				"MD5SUM": "bf28f06eb69bbd0aa2ebc2c582169d4f"},
	"DNA150": {"FILEID": "1HrFwr-VQrUHA9vdUowQtOgTCxb6IBA9u",
				"MD5SUM": "2f45b2c446012d7d2d2eeb412e3b565d"},
	"DNA250": {"FILEID": "1C-MMl-tMuTJnEkzTrt7EEIRJKB5OqZha",
				"MD5SUM": "c6830e7d612fe8c373add16371a6e752"},
	"RNA150": {"FILEID": "1JHD146DDftVLmM8yecNxjxR28v8SUtGt",
				"MD5SUM": "14b8f93e7e62b70b8229341acea6d2d9"},
	"RNA250": {"FILEID": "1c_jKpqDE8L7hZOKkiTPai53FNzYVGscp",
				"MD5SUM": "4680ba8c35a354f35c02b5e7b04255ea"},
}

def parse_args(argv = sys.argv[1:]):
	parser = argparse.ArgumentParser()

	# optional arguments
	parser.add_argument("-o", "--out-dir",
		dest="o_dir",
		metavar="PATH",
		default=".",
		help="output directory; default .")

	# required arguments:
	req_group = parser.add_argument_group("required argumnets")
	req_group.add_argument("-d",
		dest="db",
		choices=["all"] + list(FILEID_MAP.keys()),
		nargs="+",
		required=True,
		help="a list of DB names to be downloaded")

	return parser.parse_args(argv)

def main(argv = sys.argv[1:]):
	args = parse_args(argv)

	targets = args.db
	if 'all' in targets:
		targets = list(FILEID_MAP.keys())

	_ = sp.run(['mkdir', '-pv', args.o_dir])

	for model in tqdm(targets, desc = "Download models"):
		file_id = FILEID_MAP[model]["FILEID"]
		md5sum = FILEID_MAP[model]["MD5SUM"]

		# Download from Google Drive
		cmd = ["./gdown.sh", file_id, f"{args.o_dir}/{model}.tar.gz"]
		_ = sp.run(cmd)

		# Remove cookies
		cmd = ["rm", "-rf", "./cookies.txt"]
		_ = sp.run(cmd)

		# Get md5sum
		cmd = ["md5sum", f"{args.o_dir}/{model}.tar.gz"]
		res = sp.run(cmd, stdout = sp.PIPE)
		if(res.stdout.decode('utf-8').split(' ')[0] != md5sum):
			print(f"Download Fail... ({model})", file = sys.stderr)


	return 0

if __name__ == "__main__":
	exit(main())