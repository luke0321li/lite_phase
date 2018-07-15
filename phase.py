import numpy as np
import itertools
import bitarray as bta
import sys
import math
import heapq

class Diplotype:
    def __init__(self, haplotype_1, haplotype_2, k1, k2, p_1, p_2, p_12, q_1, q_2):
        self.hap_1 = haplotype_1
        self.hap_2 = haplotype_2
        self.k1 = k1
        self.k2 = k2
        self.p_1 = p_1
        self.p_2 = p_2
        self.p_12 = p_12
        self.q_1 = q_1
        self.q_2 = q_2

    def reverse(self):
        new_hap_1 = bta.bitarray(self.hap_2)
        new_hap_2 = bta.bitarray(self.hap_1)
        return Diplotype(new_hap_1, new_hap_2, self.k2, self.k1, self.p_2, self.p_1, self.p_12, self.q_2, self.q_1)


    def copy(self):
        new_hap_1 = bta.bitarray(self.hap_1)
        new_hap_2 = bta.bitarray(self.hap_2)
        return Diplotype(new_hap_1, new_hap_2, self.k1, self.k2, self.p_1, self.p_2, self.p_12, self.q_1, self.q_2)


def is_compatible(genotype, haplotype):
    """
    Checks whether a haplotype is compatible with a genotype
    """
    if len(genotype) == len(haplotype):
        for i in range(len(haplotype)):
            if haplotype[i] > genotype[i]:
                return False
        return True
    return False


def get_haplotypes(genotype):
    """
    Enumerate all compatible haplotypes of a genotype
    Input: A list of {0, 1, 2}
    Output: A list of bitarrays representing haplotypes.
    Note: haps[2n] and haps[2n + 1] are paired
    """
    # First. generate a partially filled bitarray
    # Get positions of homozygous and heterozygous sites
    hetero_pos = []
    bits = bta.bitarray(len(genotype))
    for i in range(len(genotype)):
        if genotype[i] == 1:
            hetero_pos += [i]
            bits[i] = 0
        else:
            bits[i] = (genotype[i] == 2)

    haps = []
    # Generate bitarrays corresponding to permutations
    for i in range(2 ** len(hetero_pos)):
        newbits = bits.copy()
        for j in range(len(hetero_pos)):
            if ((i >> j) & 1 == 1):
                newbits[hetero_pos[j]] = 1
        if newbits not in haps:
            haps.append(newbits)
        # if ~newbits not in haps:
        #   haps.append(~newbits)
        newbits_cmp = bits.copy()
        for j in range(len(hetero_pos)):
            if ((~i >> j) & 1 == 1):
                newbits_cmp[hetero_pos[j]] = 1
        if newbits_cmp not in haps:
            haps.append(newbits_cmp)

        if (len(haps) >= 2 ** len(hetero_pos)):
            break

    # By default the returned haplotypes are sorted in pairs
    return haps


def get_all_haplotypes(n):
    """
    Generate all 2^n haplotypes of length n
    """
    haps = []
    zeros = "0" * n
    bits = bta.bitarray(zeros)
    for i in range(2 ** n):
        newbits = bits.copy()
        for j in range(n):
            if ((i >> j) & 1 == 1):
                newbits[j] = 1
        haps.append(newbits)
    return haps


def get_unique_haplotypes(genotypes):
    """
    Given a list of genotypes, generate the library of all compatible haplotypes
    Output: hap_lib is a list of bitarrays. hap_comp is a list of lists.
    hap[n] consists of indexes of compatible haplotypes in hap_lib, for genotype n
    """
    hap_lib = []
    hap_comp = []
    for genotype in genotypes:
        haps = get_haplotypes(genotype)
        comp = []
        for hap in haps:
            if hap not in hap_lib:
                hap_lib.append(hap)
                comp.append(len(hap_lib) - 1)
            else:
                comp.append(hap_lib.index(hap))
        hap_comp.append(comp)

    return hap_lib, hap_comp


def phase_segment(genotypes, n_pops=1, q_init=None, max_iter=100, top_k=1, verbose=False):
    """
    Phase the list of given genotypes using Expectation-Maximization.
    Output: A list with each element being the form:
    [[haplotype_1, haplotype_2, population_1, population_2], ...]
    i.e. the list of the top_k diplotypes with highest probability
    and the total likelihood. Or if freqs=True, output all
    haplotypes and their frequencies.
    """
    n_indvs = len(genotypes)
    haps, comps = get_unique_haplotypes(genotypes)
    n_haps = len(haps)

    # p[h][k]: Frequency of haplotype h in population k, or P(h|k)
    # p = np.random.rand(n_haps, n_pops)
    # p = p / np.sum(p, axis=0)
    p = np.zeros((n_haps, n_pops))
    p.fill(1.0 / n_haps)

    # q[k]: Probability that a chromosome comes from population k
    q = np.random.rand(n_pops)
    if q_init is not None and len(q_init) == n_pops:
        q = np.asarray(q_init)
    q = q / np.sum(q)

    # a[i][k1][k2][h1][h2]: Probability that genotype i is explained by h1 and h2 from k1 and k2
    a = np.zeros((n_indvs, n_pops, n_pops, n_haps, n_haps))
    # Do the EM for serveral iterations
    p_prev, q_prev = p.copy(), q.copy()
    for it in range(max_iter):
        qp = q_prev * p_prev
        qpqp = np.zeros((n_pops, n_pops, n_haps, n_haps))
        sums = np.zeros(n_indvs)
        for i in range(n_indvs):
            comp_haps = comps[i]
            if len(comp_haps) == 1:
                for k1, k2 in itertools.product(range(n_pops), range(n_pops)):
                    qpqp[k1, k2, comp_haps[0], comp_haps[0]] = qp[comp_haps[0], k1] * qp[comp_haps[0], k2]
                    a[i, k1, k2, comp_haps[0], comp_haps[0]] = 1
                continue

            n_pairs = int(len(comp_haps) / 2)
            for d, k1, k2 in itertools.product(range(n_pairs), range(n_pops), range(n_pops)):
                h1 = comp_haps[2 * d]
                h2 = comp_haps[2 * d + 1]
                qpqp[k1, k2, h1, h2] = qp[h1, k1] * qp[h2, k2]
                a[i, k1, k2, h1, h2] = qpqp[k1, k2, h1, h2]
                sums[i] +=  qpqp[k1, k2, h1, h2]

            a[i, :, :, :, :] /= sums[i]

        for k in range(n_pops):
            q[k] = np.sum(a[:, k, :, :, :]) + np.sum(a[:, :, k, :, :])- np.sum(a[:, k, k, :, :])
        q = q / np.sum(q)

        for k, h in itertools.product(range(n_pops), range(n_haps)):
            p[h, k] = np.sum(a[:, k, :, h, :]) + np.sum(a[:, :, k, :, h])
        p = p / np.sum(p, axis=0)

        if (np.mean(np.abs(p - p_prev)) <= 0.001):
            break

        q_prev = q.copy()
        p_prev = p.copy()

    # Build the phased diplotypes, also calculate the likelihood
    dips = []
    qp = q * p

    for i in range(n_indvs):
        comp_haps = comps[i]
        if len(comp_haps) == 1:
            h = comp_haps[0]
            k = np.unravel_index(np.argmax(a[i, :, :, h, h], axis=None), a.shape)[0]
            dip = Diplotype(haps[h], haps[h], k, k, p[h, k], p[h, k], a[i, k, k, h, h], q[k], q[k])
            dips.append([dip])
            continue

        n_pairs = int(len(comp_haps) / 2)
        d_best, k1_best, k2_best = (0, 0, 0)
        heap = []
        l = 0.0
        for d, k1, k2 in itertools.product(range(n_pairs), range(n_pops), range(n_pops)):
            h1 = comp_haps[2 * d]
            h2 = comp_haps[2 * d + 1]
            l += qpqp[k1, k2, h1, h2]
            prob = a[i, k1, k2, h1 ,h2]
            heapq.heappush(heap, (1.0 - prob, [h1, h2, k1, k2]))

        output = []
        for num in range(top_k):
            if num >= len(heap):
                break
            raw = heapq.heappop(heap)[1]
            h1, h2, k1, k2 = raw[0], raw[1], raw[2], raw[3]
            dip = Diplotype(haps[h1], haps[h2], k1, k2, p[h1, k1], p[h2, k2], a[i, k1, k2, h1, h2], q[k1], q[k2])
            output.append(dip)
        dips.append(output)

    if not verbose:
        return dips
    return dips, haps, p


def make_window(hap_1, hap_2, size):
    output = bta.bitarray()
    output.extend(hap_1[-1 * size:])
    output.extend(hap_2[:size])
    return output


def overlapping(hap, window):
    size = int(window.length() / 2)
    return hap[-1 * size:] == window[:size]


def make_output(diplotypes):
    """
    Formats the output to stdout.
    """
    for i in range(diplotypes[0].hap_1.length()):
        out = []
        for j in range(len(diplotypes)):
            out.append('1' if diplotypes[j].hap_1[i] else '0')
            out.append('1' if diplotypes[j].hap_2[i] else '0')
        sys.stdout.write(' '.join(out))
        sys.stdout.write('\n')


def phase_full(genotypes, seg_len=8, w_size=8, n_pops=1, q_init=None, max_iter=100, n_threads=1, top_k=1, pos_file=""):
    """
    Divide-and-conquer phasing using window-EM method.
    """

    # Get genomic positions of each snp, potentially useful
    pos = []
    if pos_file != "":
        pos = np.loadtxt(pos_file, dtype=int)
        sys.stderr.write("Loaded genomic positions\n")

    # First phase each segment
    # TODO: implement multithreading because this is gonna be very slow
    geno_len = len(genotypes[0])
    sys.stderr.write("Genotype length: %d \n" % geno_len)
    sys.stderr.write("Segment length: %d \n" % seg_len)
    n_indvs = len(genotypes)
    n_segs = math.ceil(geno_len / seg_len)

    dips = []
    for n in range(n_segs):
        if n * seg_len > geno_len:
            break
        segment = genotypes[:, n * seg_len:(n + 1) * seg_len]
        if (geno_len - n * seg_len) < seg_len:
            segment = genotypes[:, n * seg_len:]
        dips.append(phase_segment(segment, n_pops, q_init, max_iter, top_k, False))
        # haps.append(hap)
        # freqs.append(p)
        if n % 100 == 0:
            sys.stderr.write("Phased segment %d of %d\n" %(n + 1, n_segs))

    # Then phase each window among segments
    # TODO: also implement multithreading
    w_len = 2 * w_size
    start = seg_len - w_size
    phased = []
    for i in range(n_indvs):
        phased.append(dips[0][i][0])

    make_output(phased)

    for n in range(n_segs - 1):
        w_start = start + n * seg_len
        w_end = w_start + w_len

        w_haps = []
        w_freqs = []

        _, w_haps, w_freqs = phase_segment(genotypes[:, w_start:w_end], 1, q_init, max_iter, top_k, True)
        for i in range(n_indvs):
            best_score = 0.0
            best_phase = None
            for k in range(len(dips[n + 1][i])):
                temp = None
                cis_w_1 = make_window(phased[i].hap_1, dips[n + 1][i][k].hap_1, w_size)
                cis_w_2 = make_window(phased[i].hap_2, dips[n + 1][i][k].hap_2, w_size)
                trans_w_1 = make_window(phased[i].hap_1, dips[n + 1][i][k].hap_2, w_size)
                trans_w_2 = make_window(phased[i].hap_2, dips[n + 1][i][k].hap_1, w_size)

                f_c1, f_c2 = w_freqs[w_haps.index(cis_w_1)], w_freqs[w_haps.index(cis_w_2)]
                f_t1, f_t2 = w_freqs[w_haps.index(trans_w_1)], w_freqs[w_haps.index(trans_w_2)]

                # phased[i].hap_1 = dips[n + 1][i][k].hap_2.copy()
                # phased[i].hap_2 = dips[n + 1][i][k].hap_1.copy()

                if f_c1 * f_c2 >= f_t1 * f_t2:
                    temp = dips[n + 1][i][k].copy()
                    # phased[i].hap_1 = dips[n + 1][i][k].hap_1.copy()
                    # phased[i].hap_2 = dips[n + 1][i][k].hap_2.copy()
                else:
                    temp = dips[n + 1][i][k].reverse()
                    # phased[i].hap_1 = dips[n + 1][i][k].hap_2.copy()
                    # phased[i].hap_2 = dips[n + 1][i][k].hap_1.copy()

                current_score = max(f_c1 * f_c2, f_t1 * f_t2) * temp.p_12
                if current_score >= best_score:
                    best_phase = temp.copy()
                    best_score = current_score
            phased[i] = best_phase.copy()

        make_output(phased)

        if n % 100 == 0:
            sys.stderr.write("Merged segment %d of %d\n" % (n + 1, n_segs))

    return phased

# TODO: implement phasing with prior on haplotype frequencies

# TEST

"""
g3 = [1, 1, 1, 2, 2, 0, 1 ,1, 1, 1, 2, 1, 1, 0 ,0, 1]
g2 = [2, 1, 1, 1, 1, 2, 1, 0, 2, 1, 1, 1, 0, 0, 1, 1]
g1 = [0, 2, 0, 1, 1, 2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0]
g4 = [1, 1, 2, 1, 2, 2, 2, 0, 0, 1, 0, 1, 2, 2, 2, 1]
g5 = [2, 2, 2, 1, 1, 1, 1, 1, 2, 0, 2, 1, 0, 0, 1, 2]
g6 = [1, 2, 2, 0, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 0]
g = [g1, g2, g3, g4, g5, g6]
g = np.asarray(g)

phase_full(g, seg_len=5, w_size=3)

print(phase_segment(g))

g1 = [1, 2, 2, 2, 0, 0, 1, 2]
g2 = [1, 1, 1, 0, 1, 2, 2, 1]
g3 = [2, 0, 0, 1, 1, 2, 0, 1]

g = [g1, g2, g3]
g = np.asarray(g)
"""
