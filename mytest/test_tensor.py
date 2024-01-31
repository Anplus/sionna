for char in "abcdefghij":
    print(char)

dict = {"x": 1, "y": 2}
for k, value in dict.items():
    print(k, value)

mylist = [1,2,3,4]
myiter = iter(mylist)
print(next(myiter))

# simplicity, maintainablitiy, reusablity, scoping
# os
import os
print(os.getcwd())
print(os.listdir())

import sionna
print(dir(sionna))
# from import, import member
# import as
import tensorflow as tf
import numpy as np
t5 = np.reshape(np.arange(18), [2, 3, 3])
print(t5)
print(tf.gather_nd(t5, indices=[[0, 0, 0], [1, 2, 1]]))
print(tf.gather_nd(t5,
                   indices=[[[0, 0], [0, 2]], [[1, 0], [1, 2]]]))
print(tf.gather_nd(t5,
                   indices=[[0, 0], [0, 2], [1, 0], [1, 2]]))
t14 = tf.constant([[-2, -7, 0],
                   [-9, 0, 1],
                   [0, -3, -8]])

t15 = tf.tensor_scatter_nd_min(t14,
                               indices=[[0, 2], [1, 1], [2, 0]],
                               updates=[-6, -5, -4])

print(t15)
