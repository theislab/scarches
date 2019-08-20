import fileinput
import sys

data_names = None

if data_names is None:
    if len(sys.argv) == 1:
        print('Enter data names followed by EOF/Ctrl-D:')
    
    data_names = []
    for line in fileinput.input():
        fields = line.rstrip().split(',')
        for f in fields:
            if f.strip() == '':
                continue
            data_names.append(f)
    print('Data names loaded')
