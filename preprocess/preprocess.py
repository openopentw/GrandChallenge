#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from os import path
import pdb
import re

import jieba

non_chi_pat = u'[^\u4e00-\u9fff]'

def preprocess(out_filename):
    out_f = open(out_filename, 'a')

    corpus_dir = "../data/training_data/subtitle_no_TC/"
    for _, dirs, corpus_files in os.walk(corpus_dir):
        for dir in dirs:
            print(dir)
            for _,_,files in os.walk(corpus_dir+'/'+dir):
                for fn in files:
                    if fn[0] == '.':
                        continue
                    print("Start %s" % fn)
                    # f = open(corpus_dir+'/'+dir+'/'+fn, 'r')
                    f = open(path.join(corpus_dir, dir, fn), 'r')
                    for line in f:
                        line = re.sub(non_chi_pat, ' ', line)

                        # word segmentation with jieba.
                        words = list(jieba.cut(line))
                        words = [w for w in words if w != ' ']
                        # TODO: deal with stop words
                        # chinese stopwords: https://github.com/stopwords-iso/stopwords-zh
                        # write to file.
                        out_f.write('%s\n' % ' '.join(words))
                    out_f.write('\n')
                    print("Done with " + fn)

if __name__ == '__main__':
    preprocess('corpus.txt')
