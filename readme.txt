Requirements:
Python3
Numpy
Bitarray

Using the default parameters (-l 8 -w 8 -p 1 -k 1) produces results for most
datasets under 40 minutes. However, using longer segments, windows and 
k produces better results (e.g. -l 10 -w 10 -k 3)


usage: main.py [-h] [-l [SEG_LEN]] [-w [WIN_LEN]] [-p [N_POPS]] [-r [RATIO]]
               [-k [TOP_K]]
               [input_file] [output_file]

positional arguments:
  input_file            Path of input genotype data file, in HAP format (http://mathgen.stats.ox.ac.uk/genetics_software/shapeit/shapeit.html#haplegsample)
  output_file           Path of output phased haplotype file, default stdout

optional arguments:
  -h, --help            show this help message and exit
  -l [SEG_LEN], --segment_length [SEG_LEN]
                        Length of genotype segment to do EM on, default 8
  -w [WIN_LEN], --window_length [WIN_LEN]
                        Length of genotype window overlapping with segments,
                        default 8
  -p [N_POPS], --populations [N_POPS]
                        Number of populations that contribute to the
                        admixture, default 1
  -r [RATIO], --guess_ratio [RATIO]
                        For two populations, a guess on the admixture ratio
                        between 0 and 1, default 0.5
  -k [TOP_K], --top_k [TOP_K]
                        When doing EM, keep the top k possible phases into
                        consideration. default 1
