from os import path

import numpy as np

# parameters
output_IDs = [
    8,
    # 9,
    # 10,
    11,
    12,
    13,
    14,
]

ans_path = './subm/'

# generate ans_path
ans_filename = str(output_IDs[0])
for ID in output_IDs[1:]:
    ans_filename += '_' + str(ID)
ans_filename += '.csv'
ans_path = path.join(ans_path, ans_filename)
print('print answer to: {}'.format(ans_path))

# import output files
outputs = np.zeros((len(output_IDs), 500, 6))
for i,ID in enumerate(output_IDs):
    output_path = path.join('output', str(ID) + '.csv')
    print(output_path)
    outputs[i] = np.genfromtxt(output_path, delimiter=',')

# get the answer from outputs
# output = outputs.sum(axis=0)
output = outputs[0] + outputs[1] + 2 * outputs[2] + outputs[3] + outputs[4]
ans = np.argmax(output, axis=1)

# print ans out
with open(ans_path, 'w') as f:
    print('id,ans', file=f)
    for i, a in enumerate(ans):
        print('{},{}'.format(i+1, a), file=f)
