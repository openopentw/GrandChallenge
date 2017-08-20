#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import pdb
import re

import jieba

non_chi_pat = u'[^\u4e00-\u9fff]'

def preprocess(out_filename):
    out_f = open(out_filename, 'a')

    corpus_dir = "pre_subtitle/pre_subtitle_no_TC/"
    for _, dirs, corpus_files in os.walk(corpus_dir):
        for dir in dirs:
            print(dir)
            for _,_,files in os.walk(corpus_dir+'/'+dir):
                for fn in files:
                    if fn[0] == '.':
                        continue
                    print("Start %s" % fn)
                    f = open(corpus_dir+'/'+dir+'/'+fn, 'r')
                    for line in f:
                        line = re.sub(non_chi_pat, ' ', line)

                        # word segmentation with jieba.
                        words = list(jieba.cut(line))
                        #words = [w for w in words if w != ' ']
                        # write to file.
                        out_f.write('%s\n' % ' '.join(words))
                    print("Done with " + fn)

if __name__ == '__main__':
    preprocess('corpus.txt')
