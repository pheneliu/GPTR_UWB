# Some user defined functions

# For printing line
from inspect import currentframe, getframeinfo

import os, shutil
import numpy as np

# Quaternion multiplication
from evo.core.transformations import quaternion_multiply as quatmult
# Conversion between quat and rotm
from evo.core.transformations import quaternion_matrix as quat2rotm_
from evo.core.transformations import quaternion_from_matrix
# Conversion between eul angles and rotm
from evo.core.transformations import euler_matrix as eul2rotm_
from evo.core.transformations import euler_from_matrix as rotm2eul_

# Coversion among rotation formulisms

def eul2rotm(e0, e1, e2):
    return eul2rotm_(e0, e1, e2, 'rzyx')[0:3, 0:3]

def eul2quat(e0, e1, e2):
    return rotm2quat(eul2rotm(e0, e1, e2))

def quat2rotm(q):
    return quat2rotm_(q)[0:3, 0:3]

def quat2eul(q):
    return rotm2eul(quat2rotm(q))

def rotm2eul(M):
    return rotm2eul_(M, 'rzyx')

def rotm2quat(M):
    return quaternion_from_matrix(M)

# Converting (x y z qx qy qz qw) into T
def pose2Tf(pose):
    q = pose[0, [6, 3, 4, 5]]
    t = np.reshape(pose[0, [0, 1, 2]], (3, 1))
    return Qt2Tf(q, t)

def pose2Rt(pose):
    pose_ = np.reshape(pose, (1, 7))
    R = quat2rotm(pose_[0, [6, 3, 4, 5]])
    t = np.reshape(pose_[0, [0, 1, 2]], (3, 1))
    return R, t

def Tf2Qt(T):
    q = rotm2quat(T[0:3, 0:3])
    t = T[0:3, 3:4]
    return q, t

def Qt2Tf(q, t):
    T = np.identity(4)
    T[0:3, 0:3] = quat2rotm(q)
    T[0:3, 3:4] = t
    return T

def Rt2Tf(R, t):
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3:4] = t
    return T

def Tf2Rt(T):
    return (T[0:3, 0:3], T[0:3, 3:4])

def tfinv(T):
    R = T[0:3, 0:3].copy()
    t = T[0:3, 3:4].copy()
    R = R.T
    t = np.dot(R, t)*(-1)
    return Rt2Tf(R, t)

def tfmult(T1, T2):
    
    R1 = T1[0:3, 0:3].copy()
    t1 = T1[0:3, 3:4].copy()

    R2 = T2[0:3, 0:3].copy()
    t2 = T2[0:3, 3:4].copy()

    R = np.dot(R1, R2)
    t = np.dot(R1, t2) + t1
    return Rt2Tf(R, t)

def resetdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=False)

if __name__ == "__main__":
    print("This is the myutil.py file.")
