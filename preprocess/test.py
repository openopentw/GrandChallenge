import jieba
import pandas as pd
import re

# parameter
test_data_path = '../data/AIFirstProblem.txt'
out_path = './test_corpus.csv'
non_chi_pat = u'[^\u4e00-\u9fff]'

# read test data
test_data = pd.read_csv(test_data_path)
test_data['dialogue'] = test_data['dialogue'].str.replace('\t', ' ')

# split the 6 option to 0 ~ 5 & drop 'id' column
test_options = test_data['options'].str.split('\t', expand=True)
test_data = test_data.drop('options', axis=1)
test_data = pd.concat([test_data, test_options], axis=1)
# test_data = test_data.values

# split words
for r in range(test_data.shape[0]):
    for c in range(1, test_data.shape[1]):
        line = re.sub(non_chi_pat, ' ', test_data.iloc[r,c])
        words = list(jieba.cut(line))
        words = [w for w in words if w != ' ']
        # TODO: deal with stop words
        test_data.iloc[r,c] = ' '.join(words)

# write to file
test_data.to_csv(out_path, sep=',', index=False)
