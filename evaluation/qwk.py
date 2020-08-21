# the implementation of quadratic weighted kappa
import numpy as np
import sys
import os
from statistics import mean

def get_weight_matrix(grade_scale):
    if grade_scale < 2:
        print('The grade_scale is supposed to be greater than 1.')
        return
    m = np.zeros((grade_scale, grade_scale))
    for row in range(grade_scale):
        for col in range(grade_scale):
            m[row,col] = (row-col)**2 / (grade_scale-1)**2
    return m


def get_expected_matrix(grades,preds):
    if sum(grades) != sum(preds):
        print('The number of elements from confusion matrix is not equal.')
        return
    e = np.zeros((len(grades),len(preds)))
    for row, grade in enumerate(grades):
        for col, pred in enumerate(preds):
            e[row,col] = grade * pred

    # do normalization such that the number is equal to its original counts
    c = sum(grades)
    t = sum([sum(e[i,:]) for i in range(len(grades))])
    co = c/t
    for row in range(len(grades)):
        for col in range(len(grades)):
            e[row,col] = e[row,col] * co
    return e

def matrix_production(m,n): #m and n have same size
    result = 0
    rows, cols = m.shape
    for row in range(rows):
        for col in range(cols):
            result += m[row, col] * n[row, col]
    return result

def compute_qwk(filename):
    with open(filename, 'r') as f:
        rows = f.readlines()[1:]
    O = np.zeros((len(rows),len(rows)))
    for row, grade in enumerate(rows):
        for col, pred in enumerate(grade.split(',')[1:]):
            O[row,col] = int(pred.strip())
    grades = [sum(O[:,i]) for i in range(len(rows))]
    preds = [sum(O[i,:]) for i in range(len(rows))]
    E = get_expected_matrix(grades, preds)

    M = get_weight_matrix(len(rows))

    k = round(1 - (matrix_production(O,M)/matrix_production(E,M)), 2)

    acc = round(sum([O[i,i] for i in range(len(rows))]) / sum(grades),4)
    #logfile = '-'.join(filename.split('-')[:2]) + '-params-log.csv'
    #with open(logfile) as f:
    #    test = f.readlines()[1].split(',')[:2]

    #print('\t'.join(test))
    return filename.split('/')[-1].split('_')[1], k, acc
    print(filename.split('/')[-1].split('_')[1],'\t', k,'\t', acc)

def avg_confusion_matrix(path,files,bestn=None):
    m = np.zeros((4,4))
    for f in files:
        if 'test' in f:
            if bestn:
                if str(bestn) != f.split('/')[-1].split('_')[1]:
                    continue
            with open(path+'/'+f,'r') as fi:
                rows = fi.readlines()[1:]
            for row, grade in enumerate(rows):
                for col, pred in enumerate(grade.split(',')[1:]):
                    m[row,col] += int(pred.strip())
    if not bestn:
        m = m/10
    for row in np.flip(m):
        row = np.concatenate((row, np.zeros(1)+sum(row)))
        print('\t&'.join([str(round(i,2)) for i in row]))
    print('\t&'.join([str(round(m[:,i].sum(),2)) for i in range(4)]))


def main(filename):
    #compute_qwk(filename)
    dir_path = sys.argv[1]
    files = os.listdir(dir_path)
    re = []
    avg_confusion_matrix(dir_path, files)
    for f in files:
        if 'test' in f:
            re.append(compute_qwk(dir_path+'/'+f))
    re.sort()
    re.append(re.pop(1))
    ks = [i[1] for i in re]
    bestn = ks.index(max(ks))+1
    print(bestn)
    avg_confusion_matrix(dir_path,files,bestn)
    out = ''
    for i in re:
        out += str(i[1]) + '('+str(round(i[2],2))+')'
        out += '\t'
    out += ''.join([str(round(mean([i[1] for i in re]),2)),
                     '('+str(round(mean([i[2] for i in re]),2))+')'])
    print(out)
if __name__ == '__main__':
    main(sys.argv[1])




