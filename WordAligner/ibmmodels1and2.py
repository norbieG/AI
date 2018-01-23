#!/usr/bin/env python

#implementation of IBM models 1 and 2



import optparse
import sys
from collections import defaultdict
from math import log
import matplotlib.pyplot as plt

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--num_iterations", dest="num_iters", default=5, type="int", help="Number of iterations to use for EM optimization")
(opts, _) = optparser.parse_args()

bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]

f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
fe_align_count = defaultdict(int)
e_align_count = defaultdict(int)

w_count = defaultdict(int)
prob = defaultdict(float)
dist_param = defaultdict(float)

# MODEL 1 PART #
for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
        for e_j in set(e):
            prob[(f_i, e_j)] = 1.0 / len(e)

for iter in range(5):
    # Initialize all counts to 0
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            for e_j in set(e):
                fe_count[(f_i, e_j)] = 0
        for e_i in set(e):
            e_count[e_i] = 0

    # Compute (E)xpected counts
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            z = 0
            for e_j in set(e):
                z += prob[(f_i, e_j)]

            for e_j in set(e):
                c = prob[(f_i, e_j)] / z
                fe_count[(f_i, e_j)] += c
                e_count[e_j] += c

    # Normalize expected counts, (M)aximization step for translation probabilities
    for f_i, e_j in fe_count:
        prob[(f_i, e_j)] = fe_count[(f_i, e_j)] / e_count[e_j]

# MODEL 2 #
# Initialize distortion parameter
for (n, (f, e)) in enumerate(bitext):
    for (i, f_i) in enumerate(f):
        for (j, e_j) in enumerate(e):
            dist_param[(i, j, len(f), len(e))] = 1.0 / len(e)

iter = 0
while iter < 5:
    for (n, (f, e)) in enumerate(bitext):
        for (i, f_i) in enumerate(f):
            for (j, e_j) in enumerate(e):
                fe_count[(f_i, e_j)] = 0
                fe_align_count[(i, j, len(f), len(e))] = 0
        for (j, e_j) in enumerate(e):
            e_count[e_j] = 0
            e_align_count[(j, len(f), len(e))] = 0
    for (n, (f, e)) in enumerate(bitext):
        len_f = len(f)
        len_e = len(e)
        for (i, f_i) in enumerate(f):
            z = 0
            for (j, e_j) in enumerate(e):
                z += prob[(f_i, e_j)] * dist_param[(i, j, len_f, len_e)]
            for (j, e_j) in enumerate(e):
                c = prob[(f_i, e_j)] * dist_param[(i, j, len_f, len_e)] / z
                fe_count[(f_i, e_j)] += c
                e_count[e_j] += c
                fe_align_count[(i, j, len_f, len_e)] += c
                e_align_count[(j, len_f, len_e)] += c
    for f_i, e_j in fe_count:
        prob[(f_i, e_j)] = fe_count[(f_i, e_j)] / e_count[e_j]
    for (n, (f, e)) in enumerate(bitext):
        for (i, f_i) in enumerate(f):
            for (j, e_j) in enumerate(e):
                dist_param[(i, j, len(f), len(e))] = fe_align_count[(i, j, len(f), len(e))] / e_align_count[(j, len(f), len(e))]
    iter += 1

total_prob = 0
for (f, e) in bitext:
    sentence_prob = 0
    for (i, f_i) in enumerate(f):
        best_prob = 0
        best_j = -1
        t_prob = 0
        for (j, e_j) in enumerate(e):
            if prob[(f_i, e_j)] * dist_param[(i, j, len(f), len(e))] > best_prob:
                best_prob = prob[(f_i, e_j)] * dist_param[(i, j, len(f), len(e))]
                best_j = j
                ef = f_i
                en = e_j
            t_prob += prob[(f_i, e_j)] * dist_param[(i, j, len(f), len(e))]
        sys.stdout.write("%i-%i " % (i, best_j))
        sentence_prob += log(t_prob, 10)
    sys.stdout.write("\n")
    total_prob += sentence_prob
#print total_prob/opts.num_sents
