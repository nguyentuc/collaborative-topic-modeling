import string
import numpy as np
import sys,getopt
import math
import os

global dataname
global NTest
dataname= ''
NN = 10

try:

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hd:n:", ["dname=","ntest="])
except getopt.GetoptError:
    print 'test.py -d <dataname> -n <ntest>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'test.py -d <dataname>'
        sys.exit()
    elif opt in ("-d", "--dname"):
        dataname = arg


print "Data name : %s"%(dataname)
if dataname == '':
    print "Missing dataname"
    exit(-1)

pathOut = "results/%s/statslv.txt"%(dataname)
fIn = open("results/%s/resultlv.txt"%(dataname),"r")
fOut = open(pathOut,"w")


dictPre = dict()
dictRe = dict()
listKey = list()
dictCount = dict()
isIn = False
while True:
    line = fIn.readline()
    if line == "":
        break

    key = line.strip()
    if(key not in listKey):
        ar = np.ndarray(NN,dtype=float)
        ar.fill(0)
        dictPre[key] = ar
        ar = np.ndarray(NN, dtype=float)
        ar.fill(0)
        dictRe[key] = ar
        listKey.append(key)
        dictCount[key] = 0
    dictCount[key] += 1
    arPre = dictPre[key]
    arRec = dictRe[key]
    for i in xrange(NN):

        line = fIn.readline()
        line = line.strip()
        parts = string.split(line," ")
        pre  = float(parts[0])
        rec = float(parts[1])
        arPre[i] += pre
        arRec[i] += rec


for key in listKey:
    dictPre[key] /= dictCount[key]
    dictRe[key] /= dictCount[key]



fOut.write("Precision:\n")
for key in listKey:
    fOut.write("%s"%key)
    for i in xrange(NN):
        fOut.write(" %2.5f"%dictPre[key][i])
    fOut.write("\n")


fOut.write("Recall:\n")
for key in listKey:
    fOut.write("%s:"%key)
    for i in xrange(NN):
        fOut.write(" %2.5f"%dictRe[key][i])
    fOut.write("\n")
fIn.close()
fOut.close()
os.system("cat %s"%pathOut)

print dictCount[listKey[0]]