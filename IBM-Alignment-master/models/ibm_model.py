import json
import numpy as np
from itertools import permutations

def em_algorithm(data, iterations=30):
    sentences = []
    fr_lex = []
    en_lex = []
    for pair in data:
        en_sent = pair['en'].split()
        fr_sent = pair['fr'].split()
        sentences.append((en_sent, fr_sent))
        for word in en_sent:
            en_lex.append(word)
        for word in fr_sent:
            fr_lex.append(word)
    en_lex = list(set(en_lex))
    fr_lex = list(set(fr_lex))
    L = len(en_lex)
    M = len(fr_lex)

    t = np.full((L, M), 1/L)
    for _ in range(iterations):
        count = np.zeros((L, M))
        total = np.zeros(M)
        for pair in sentences:
            l = len(pair[0])
            m = len(pair[1])
            s_total = np.zeros(l)
            for i in range(l):
                s_total[i] = 0
                for j in range(m):
                    e = en_lex.index(pair[0][i])
                    f = fr_lex.index(pair[1][j])
                    s_total[i] += t[e][f]
            for i in range(l):
                for j in range(m):
                    e = en_lex.index(pair[0][i])
                    f = fr_lex.index(pair[1][j])
                    count[e][f] += t[e][f]/s_total[i]
                    total[f] += t[e][f]/s_total[i]
        for j in range(m):
            for i in range(l):
                e = en_lex.index(pair[0][i])
                f = fr_lex.index(pair[1][j])
                t[e][f] = count[e][f]/total[f]
    return en_lex, fr_lex, t

def alignment(data, en_lex, fr_lex, t, epsilon=1):
    sentences = []
    for pair in data:
        en_sent = pair['en'].split()
        fr_sent = pair['fr'].split()
        sentences.append((en_sent, fr_sent))

    for pair in sentences:
        l = len(pair[0])
        m = len(pair[1])
        max_p = 0
        for perm in permutations(pair[1]):
            A = list(zip(pair[0], perm))
            p = epsilon/(m**l)
            for a in A:
                e = en_lex.index(a[0])
                f = fr_lex.index(a[1])
                p = p * t[e][f]
            if p > max_p:
                max_p = p
                best_A = A
        print(best_A)
