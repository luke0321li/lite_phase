#!/usr/bin/env python3

import os, sys
import numpy as np
import argparse as ap
import phase
import time

def main():
    parser = ap.ArgumentParser()
    parser.add_argument("input_file", help="Path of input genotype data file", nargs='?')
    parser.add_argument("output_file", help="Path of output phased haplotype file, default stdout", \
                        nargs='?', default="")
    parser.add_argument("-l", "--segment_length", help="Length of genotype segment to do EM on, default 8", \
                        nargs='?', type=int, default=8, const=8, dest="seg_len")
    parser.add_argument("-w", "--window_length", help="Length of genotype window overlapping with segments, \
                        default 8", nargs='?', type=int, default=8, const=8, dest="win_len")
    # parser.add_argument("-r", "--repeats", help="How many times should the EM be repeated for each segment and window, \
    #                    default 1", nargs='?', type=int, default=1, const=1, dest="repeats")
    parser.add_argument("-p", "--populations", help="Number of populations that contribute to the admixture, \
                        default 1", nargs='?', type=int, default=1, const=1, dest="n_pops")
    parser.add_argument("-r", "--guess_ratio", help="For two populations, a guess on the admixture ratio \
                        between 0 and 1, default 0.5", nargs='?', type=float, default=0.5, const=0.5, dest="ratio")
    parser.add_argument("-k", "--top_k", help="When doing EM, keep the top k possible phases into consideration. \
                        default 1", nargs='?', type=int, default=1, const=1, dest="top_k")


    args = parser.parse_args()
    genotypes = np.loadtxt(args.input_file, dtype=int, unpack=True)
    sys.stderr.write("Loaded genotypes...: %d SNPS detected.\n" % (genotypes.shape[1]))
    if args.output_file != "":
        sys.stdout = open(args.output_file, 'w')

    start = time.time()
    if args.n_pops == 2:
        phase.phase_full(genotypes, seg_len=args.seg_len, w_size=int(args.win_len / 2), n_pops=2, \
        q_init=[args.ratio, 1.0 - args.ratio], top_k=args.top_k)
    else:
        phase.phase_full(genotypes, seg_len=args.seg_len, w_size= int(args.win_len / 2), n_pops=args.n_pops, \
        top_k=args.top_k)
    end = time.time()
    sys.stderr.write("Total runtime: %.2f seconds" % (end - start))

if __name__ == '__main__':
    main()
