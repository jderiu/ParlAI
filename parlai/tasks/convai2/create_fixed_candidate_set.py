import os

dpath = os.path.join('data', 'ConvAI2', 'valid_none_original_no_cands.txt')
data_file = open(dpath, 'rt', encoding='utf-8').readlines()

opath = os.path.join('data', 'ConvAI2', 'candidates.txt')
ofile = open(opath,  'wt', encoding='utf-8')

for line in data_file:
    line = line.replace('\n', '')
    spline = line.split('\t')
    line0 = ' '.join(spline[0].split()[1:])
    line1 = spline[1]
    ofile.write(line0 + '\n')
    ofile.write(line1 + '\n')

ofile.close()