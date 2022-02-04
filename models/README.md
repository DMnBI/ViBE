## Models directory

This is a directory to save models. You can check that models have been downloaded properly using the give MD5 checksum list.  
The pre-trained models could be downloaded through the links as follow:  

**Pre-trained model**

* [pre-trained](https://drive.google.com/file/d/100EITt7ZmyjkBl_X1kJ83nfV5jpK_ED1/view?usp=sharing)

**Domain-level classifier**

* [BPDR.150bp](https://drive.google.com/file/d/1nSTwkvfeJ5VTs2__FOIVW9IO-L8iQZid/view?usp=sharing)
* [BPDR.250bp](https://drive.google.com/file/d/1WdawuAiz1E4CYwrtjvd24dNFHUjns9ZZ/view?usp=sharing)

**Order-level classifiers**

* [DNA.150bp](https://drive.google.com/file/d/1HrFwr-VQrUHA9vdUowQtOgTCxb6IBA9u/view?usp=sharing)
* [DNA.250bp](https://drive.google.com/file/d/1C-MMl-tMuTJnEkzTrt7EEIRJKB5OqZha/view?usp=sharing)
* [RNA.150bp](https://drive.google.com/file/d/1JHD146DDftVLmM8yecNxjxR28v8SUtGt/view?usp=sharing)
* [RNA.250bp](https://drive.google.com/file/d/1c_jKpqDE8L7hZOKkiTPai53FNzYVGscp/view?usp=sharing)

or using `download_models.py` script in the `scripts` directory.

```
chmod u+x scripts/gdown.sh
python scripts/download_models.py -d all -o models
```
