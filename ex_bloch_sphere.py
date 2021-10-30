from qutip import *
import matplotlib.pyplot as plt
import math

b=Bloch()
pnt=[1/math.sqrt(3),1/math.sqrt(3),1/math.sqrt(3)]
b.add_points(pnt)
#b.show()
vec=[0,1,0]
#vec=[0,0.707,0.707]
b.add_vectors(vec)
#b.show()
up=basis(2,0)
b.add_states(up)
b.show()
