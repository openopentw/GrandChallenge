# -*- coding: utf-8 -*-

import jieba
import json
import os
import pdb
import re

from os import path

# parameters
out_filename = 'corpus_with_TC.txt'
corpus_dir = '../data/training_data/subtitle_with_TC'
stopwords_files = [
    './stopwords.txt',
    # './long_stopwords.txt',
]
non_chi_pat = u'[^\u4e00-\u9fff]'

# load stopwords
stopwords = []
for stopwords_file in stopwords_files:
    with open(stopwords_file, 'r', encoding='utf8') as f:
        stopwords += f.read().splitlines()

def split_TCs(line):
    if '\t' in line:
        sep = '\t'
        words = line.split('\t')
        if len(words) >= 3:
            index = 3
        elif len(words) == 2:
            index = 2
    else:   # ' '
        sep = ' '
        words = line.split(' ')
        if len(words) >= 3:
            index = 3
        elif len(words) == 2:
            index = 2
    return sep, index

# load files and write to one file
out_f = open(out_filename, 'a')
for _, dirs, corpus_files in os.walk(corpus_dir):
    for dir in dirs:
        print(dir)
        for _,_,files in os.walk(path.join(corpus_dir, dir)):
            for fn in files:
                if fn[0] == '.':
                    continue
                print('Start %s' % fn)

                with open(path.join(corpus_dir, dir, fn), 'r') as f:
                    for i,line in enumerate(f):
                        if i == 0:
                            # recognize sep ('\t' or ' ')
                            sep, index = split_TCs(line)
                        # split tab / blank & take the final words
                        line = line.split(sep)
                        if len(line) == index:
                            line = line[index - 1]
                        else:
                            line = ''
                        line = re.sub(non_chi_pat, ' ', line)
                        words = list(jieba.cut(line))
                        words = [w for w in words if w != ' ']
                        out_f.write('%s\n' % ' '.join(words))
                out_f.write('\n')

                print('Done with ' + fn)
out_f.close()
