import string
import numpy as np
import sys,getopt

global dataname
global NTest
dataname= ''
NTest = 2000

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
    elif opt in ("-n", "--ntest"):
        NTest = arg

print "Data name : %s"%(dataname)
print "NTest "+NTest
if dataname == '':
    print "Missing dataname"
    exit(-1)

MAXTOP = 100

pathUserTest = "../../Data/%s/users_test.dat"%dataname
pathUserTrain = "../../Data/%s/users_train.dat"%dataname

pathFinalU = "%s/final-U.dat"%dataname
pathFinalV = "%s/final-V.dat"%dataname

fUserTest = open(pathUserTest,"r")
fUserTrain = open(pathUserTrain,"r")

matU = np.loadtxt(pathFinalU,dtype=float)
matV = np.loadtxt(pathFinalV,dtype=float)
NU = matU.shape[0]
NV = matV.shape[0]
if(NTest == "all"):
    NTest = NU - 1
else:
    NTest = int(NTest)
print "%s %s %s"%(NU,NV,matU.shape[1])
matV = matV.transpose()

listUserTrains = list()
listUserTests = list()
for i in xrange(NU):
    listUserTrains.append(list())
for i in xrange(NU):
    listUserTests.append(list())

print "Loading user histories"
ic = 0
it = listUserTrains.__iter__()

while True:
    line = fUserTrain.readline()
    if line == "":
        break
    line = line.strip()
    line = string.split(line," ")
    cnt = int(line[0])
    ls = it.next()
    for i in xrange(cnt):
        ls.append(int(line[i+1]))
    ic+= 1
    #if(ic%100==0):
    #    print "\r%s" %ic
fUserTrain.close()

print "Loading test "
ic = 0
it = listUserTests.__iter__()
while True:
    line = fUserTest.readline()
    if line == "":
        break

    line = line.strip()
    line2 = string.split(line," ")
    cnt = int(line2[0])
    ls = it.next()

    for i in xrange(cnt):
        ls.append(int(line2[i+1]))
    ic += 1
    #if (ic%100==0):
    #    print "\r%s"%ic
fUserTest.close()

def evalForSingle(fu,trainu,testu):
    nTrain = len(trainu)
    nTest = len(testu)
    re = np.dot(fu,matV)
    args = re.argsort()
    ic = 0
    ix  = 0
    listRe = np.ndarray(MAXTOP,dtype=int)

    while ic < MAXTOP:
        item =  args[NV-1-ix]
        ix += 1
        if item in trainu:
            continue
        listRe[ic] = item
        ic += 1

    pre = np.ndarray(10,dtype=float)
    pre.fill(0)

    rec = np.ndarray(10,dtype=float)
    rec.fill(0)

    for i in xrange(10):
        for j in xrange(10):
            if listRe[i*10+j] in testu:
                for kk in xrange(i,10):
                    pre[kk] += 1
                    rec[kk] += 1
    if nTest == 0:
        print "Error"
        exit(-1)
    for i in xrange(10):
        pre[i] /= (i+1)*10
        rec[i] /= nTest
    return (pre,rec)
def getForSingle(fu,trainu,testu):
    nTrain = len(trainu)
    nTest = len(testu)
    re = np.dot(fu,matV)
    args = re.argsort()
    ic = 0
    ix  = 0
    listRe = np.ndarray(MAXTOP,dtype=int)

    while ic < MAXTOP:
        item =  args[NV-1-ix]
        ix += 1
        if item in trainu:
            continue
        listRe[ic] = item
        ic += 1

    return listRe
def recommend(id):
    utrain = listUserTrains[id]
    utest = listUserTests[id]
    re = getForSingle(matU[id,:],utrain,utest)
    s = ""
    for i in xrange(10):
        s = "%s %s"%(s,re[i])
    print s

def recommend2(id):
    trainu = listUserTrains[id]
    testu = listUserTests[id]
    nTrain = len(trainu)
    nTest = len(testu)
    fu = matU[id,:]
    re = np.dot(fu, matV)
    args = re.argsort()
    ic = 0
    ix = 0
    listRe = np.ndarray(MAXTOP, dtype=int)

    while ic < MAXTOP:
        item = args[NV - 1 - ix]
        ix += 1
        if item in trainu:
            continue
        listRe[ic] = item
        ic += 1

    pre = np.ndarray(10, dtype=float)
    pre.fill(0)

    rec = np.ndarray(10, dtype=float)
    rec.fill(0)

    for i in xrange(10):
        for j in xrange(10):
            if listRe[i * 10 + j] in testu:

                for kk in xrange(i, 10):
                    pre[kk] += 1
                    rec[kk] += 1
    if nTest == 0:
        print "Error"
        exit(-1)
    for i in xrange(10):
        #pre[i] = pre[i]* 1.0/ ((i + 1) * 10)
        #rec[i] = rec[i]*1.0/nTest
        print "%s %s"%(pre[i],rec[i])

def eval():
    itTrain = listUserTrains.__iter__()
    itTest = listUserTests.__iter__()

    itest = 0
    ic = 0
    r = NTest * 1.0 / NU
    pre = np.ndarray(10,dtype=float)
    rec = np.ndarray(10,dtype=float)
    pre.fill(0)
    rec.fill(0)

    while ic < NU and itest<NTest:
        utrain = itTrain.next()
        utest = itTest.next()

        fu = matU[ic,:]
        ic += 1
        if (len(utest) < 1):
            continue
        rd = np.random.random()

        if rd < r:
            itest += 1
            (prex,recx) = evalForSingle(fu,utrain,utest)
            pre = np.add(prex,pre)
            rec = np.add(recx,rec)

    print itest
    pre/= itest
    rec/=itest

    for i in xrange(10):
        print "Top %s pre: %s recal %s"%((i+1)*10,pre[i],rec[i])



eval()
#recommend2(10)






