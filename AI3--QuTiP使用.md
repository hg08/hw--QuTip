QuTiP的文档：



http://qutip.org/docs/latest/guide/guide-overview.html



QuTiP的例子：

<http://qutip.org/tutorials.html>



API documents:

http://qutip.org/docs/latest/apidoc/apidoc.html#apidoc



# Basic Operations on Quantum Objects

http://qutip.org/docs/latest/guide/guide-basics.html



导入qutip的方法

```Python
from qutip import *
```

这将导入用户可用的所有函数．函数的调用格式为 qutip.module.function() ．有了上句代码，我们就导入了所有module. 因此只需要写function()就可以调用函数了．



导入我们需要的模块：　NumPy , matplotlib.pyplot.

```python
import numpy as np
import matplotlib.pyplot as plt
```



## The quantum object class

经典力学里的变量用**数**表示；量子世界的变量用**算符**表示．



量子对象的三要素：dimensions, shape, and data． 例如，我们创建一个空量子对象．它的三要素如下：

```python
>>> qt.Qobj()
Quantum object: dims = [[1], [1]], shape = (1, 1), type = bra
Qobj data =
[[0.]]
```

注意：　The key difference between classical and quantum mechanics lies in the use of operators instead of numbers as variables．



We can create a Qobj with a user defined data set by passing a list or array of data into the Qobj:

```
>>> qt.Qobj([[0],[1],[2]])
Quantum object: dims = [[3], [1]], shape = (3, 1), type = ket
Qobj data =
[[0.]
 [1.]
 [2.]]
 
```

注意dims和 shape如何根据输入数据之变化而变化. Although dims and shape appear to have the same function, the difference will become quite clear in the section on tensor products and partial traces.



### States and operators

手动地为每一个量子对象指定数据是低效的. Even more so when most objects correspond to commonly used types such as the ladder operators of a harmonic oscillator, the Pauli spin operators for a two-level system, or state vectors such as Fock states. 因此, 对于各种态，QuTiP 包含了预先定义好的对象:



各种态:

| States                                   | Command (# means optional) | Inputs                                                       |
| ---------------------------------------- | -------------------------- | ------------------------------------------------------------ |
| Fock state ket vector　(Fock态右矢量)    | `basis(N,#m)`/`fock(N,#m)` | N = number of levels in Hilbert space, m = level containing excitation (0 if no m given) |
| Fock密度矩阵(outer product of basis)     | `fock_dm(N,#p)`            | same as basis(N,m) / fock(N,m)                               |
| Coherent state(相干态)矢量               | `coherent(N,alpha)`        | alpha = complex number (eigenvalue) for requested coherent state |
| Coherent density matrix (outer product)  | `coherent_dm(N,alpha)`     | same as coherent(N,alpha)                                    |
| Thermal density matrix (for n particles) | `thermal_dm(N,n)`          | n = particle number expectation value                        |

各种算符:

| Operators                           | Command (# means optional) | Inputs                                                       |
| ----------------------------------- | -------------------------- | ------------------------------------------------------------ |
| Charge operator                     | `charge(N,M=-N)`           | Diagonal operator with entries from M..0..N.                 |
| Commutator                          | `commutator(A, B, kind)`   | Kind = ‘normal’ or ‘anti’.                                   |
| Diagonals operator                  | `qdiags(N)`                | Quantum object created from arrays of diagonals at given offsets. |
| Displacement operator (Single-mode) | `displace(N,alpha)`        | N=number of levels in Hilbert space, alpha = complex displacement amplitude. |
| Higher spin operators               | `jmat(j,#s)`               | j = integer or half-integer representing spin, s = ‘x’, ‘y’, ‘z’, ‘+’, or ‘-‘ |
| Identity                            | `qeye(N)`                  | N = number of levels in Hilbert space.                       |
| Lowering (destruction) operator     | `destroy(N)`               | same as above                                                |
| Momentum operator                   | `momentum(N)`              | same as above                                                |
| Number operator                     | `num(N)`                   | same as above                                                |
| Phase operator (Single-mode)        | `phase(N, phi0)`           | Single-mode Pegg-Barnett phase operator with ref phase phi0. |
| Position operator                   | `position(N)`              | same as above                                                |
| Raising (creation) operator         | `create(N)`                | same as above                                                |
| Squeezing operator (Single-mode)    | `squeeze(N, sp)`           | N=number of levels in Hilbert space, sp = squeezing parameter. |
| Squeezing operator (Generalized)    | `squeezing(q1, q2, sp)`    | q1,q2 = Quantum operators (Qobj) sp = squeezing parameter.   |
| Sigma-X                             | `sigmax()`                 |                                                              |
| Sigma-Y                             | `sigmay()`                 |                                                              |
| Sigma-Z                             | `sigmaz()`                 |                                                              |
| Sigma plus                          | `sigmap()`                 |                                                              |
| Sigma minus                         | `sigmam()`                 |                                                              |
| Tunneling operator                  | `tunneling(N,m)`           | Tunneling operator with elements of the form \|N><N+m\|+\|N+m><N\|\|N><N+m\|+\|N+m><N\|. |

As an example, we give the output for a few of these functions:





### Qobj attributes

We have seen that a quantum object has several internal attributes, such as **data, dims, and shape**. 数据，维度和形状.

These can be accessed in the following way:



## Functions operating on Qobj class

Like attributes, the quantum object class has defined functions (methods) that operate on `Qobj`class instances. For a general quantum object `Q`:

| Function         | Command                     | Description                                                  |
| ---------------- | --------------------------- | ------------------------------------------------------------ |
| Check Hermicity  | `Q.check_herm()`            | Check if quantum object is Hermitian                         |
| Conjugate        | `Q.conj()`                  | Conjugate of quantum object.                                 |
| Cosine           | `Q.cosm()`                  | Cosine of quantum object.                                    |
| Dagger (adjoint) | `Q.dag()`                   | Returns adjoint (dagger) of object.                          |
| Diagonal         | `Q.diag()`                  | Returns the diagonal elements.                               |
| Diamond Norm     | `Q.dnorm()`                 | Returns the diamond norm.                                    |
| Eigenenergies    | `Q.eigenenergies()`         | Eigenenergies (values) of operator.                          |
| Eigenstates      | `Q.eigenstates()`           | Returns eigenvalues and eigenvectors.                        |
| Eliminate States | `Q.eliminate_states(inds)`  | Returns quantum object with states in list inds removed.     |
| Exponential      | `Q.expm()`                  | Matrix exponential of operator.                              |
| Extract States   | `Q.extract_states(inds)`    | Qobj with states listed in inds only.                        |
| Full             | `Q.full()`                  | Returns full (not sparse) array of Q’s data.                 |
| Groundstate      | `Q.groundstate()`           | Eigenval & eigket of Qobj groundstate.                       |
| Matrix Element   | `Q.matrix_element(bra,ket)` | Matrix element <bra\|Q\|ket>                                 |
| Norm             | `Q.norm()`                  | Returns L2 norm for states, trace norm for operators.        |
| Overlap          | `Q.overlap(state)`          | Overlap between current Qobj and a given state.              |
| Partial Trace    | `Q.ptrace(sel)`             | Partial trace returning components selected using ‘sel’ parameter. |
| Permute          | `Q.permute(order)`          | Permutes the tensor structure of a composite object in the given order. |
| Projector        | `Q.proj()`                  | Form projector operator from given ket or bra vector.        |
| Sine             | `Q.sinm()`                  | Sine of quantum operator.                                    |
| Sqrt             | `Q.sqrtm()`                 | Matrix sqrt of operator.                                     |
| Tidyup           | `Q.tidyup()`                | Removes small elements from Qobj.                            |
| Trace            | `Q.tr()`                    | Returns trace of quantum object.                             |
| Transform        | `Q.transform(inpt)`         | A basis transformation defined by matrix or list of kets ‘inpt’ . |
| Transpose        | `Q.trans()`                 | Transpose of quantum object.                                 |
| Truncate Neg     | `Q.trunc_neg()`             | Truncates negative eigenvalues                               |
| Unit             | `Q.unit()`                  | Returns normalized (unit) vector Q/Q.norm().                 |

```python
In [24]: basis(5, 3)
Out[24]: 
Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
Qobj data =
[[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 0.]]
```



# 参考习题：

https://cs.uwaterloo.ca/~watrous/CS766/



# 附录：基本概念

**quantum channel**:  a map is called a quantum channel if it always maps valid states to valid states. Formally, a map is a channel if it is both completely positive and trace preserving.