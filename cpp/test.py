import numpy as np
import dstl
a=np.array([[1,2,3],[4,5,6]])
#a=np.array([1,2,3], dtype='float64')
a=np.array([[1,2,3],[4,5,6]], dtype='float64')

b=dstl.stlm(a)
#print(a)
