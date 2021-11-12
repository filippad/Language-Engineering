import numpy as np 
import matplotlib.pyplot as plt

scores = [3, 2, 1, 0, 0, 0, 3, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0 ,0 ,0, 0, 0, 3, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

nRetrieve = len(scores)

print(nRetrieve)

def precicion (nRelevance):
    return nRelevance/nRetrieve

def recall (nRelevance):
    return nRelevance/100

precs = []
recs = []
toP = [10, 20, 30, 40, 50]

for i in range(1,nRetrieve+1,1):
    l = scores[0:i]
    nRelevance = sum(f > 0 for f in l)
    precs.append(nRelevance/(len(l)))
    recs.append(nRelevance/100)
    if(i in toP):
        print('Top ', i, ' docs, precision: ', nRelevance/(len(l)), ' recall: ' , nRelevance/100)

inds = np.arange(0, nRetrieve, 1)

#plt.plot(inds, precs, 'b-', label='precision')
#plt.plot(inds, recs, 'r-', label='recall')
plt.plot(recs, precs, 'b-')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc="upper right")
plt.show()