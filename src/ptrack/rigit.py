"""


Based on source code by Shuo Han.
https://github.com/hanshuo/ros_rigit.git

SCL; 4 Jun 2014
"""

import numpy as np
import numpy.linalg
from numpy import *

import time
import itertools


def err2(body, world, R=None, T=None):
    assert body.shape[1] == world.shape[1]
    if R is None:
        R = np.eye(2)
    if T is None:
        T = np.zeros(2).reshape((2,1))
    total = 0.0
    for i in range(body.shape[1]):
        x = world[:,i]-(np.dot(R, body[:,i])+T)
        total += np.dot(x, x)
    return total


def rigit2(body, world):
    assert shape(world)[0] == 2 and shape(body)[0] == 2, \
        'Only accepts points in R^2. Maybe try transposing the data matrices?'
    assert shape(world)[1] == shape(body)[1], \
        'Number of world points and body points does not match'
    npoints = shape(world)[1]

    centroid_body = mean(body, 1)
    centroid_world = mean(world, 1)
    T = centroid_world - centroid_body

    body_nc = body - centroid_body[:,newaxis]
    world_nc = world - centroid_world[:,newaxis]

    X1 = np.sum(world_nc*body)
    X2 = np.sum(world_nc[[1,0],:]*(body*np.outer(np.array([1,-1]), np.ones(npoints))))

    theta = numpy.arctan2(X2, X1)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    T += np.dot(np.eye(2) - R, centroid_body)

    return R, T, err2(body, world, R, T)


def rigit(body, world):
    assert shape(world)[0] == 3 and shape(body)[0] == 3, \
        'Only accepts points in R^3. Maybe try transposing the data matrices?'
    assert shape(world)[1] == shape(body)[1], \
        'Number of world points and body points does not match'
    npoints = shape(world)[1]

    centroid_body = mean(body, 1)
    centroid_world = mean(world, 1)

    body_nc = body - centroid_body[:,newaxis]
    world_nc = world - centroid_world[:,newaxis]

    a_all = vstack((zeros(npoints), body_nc)).T
    b_all = vstack((zeros(npoints), world_nc)).T

    Ma_all = array([[[a[0], -a[1], -a[2], -a[3]],
                     [a[1],  a[0],  a[3], -a[2]],
                     [a[2], -a[3],  a[0],  a[1]],
                     [a[3],  a[2], -a[1],  a[0]]]
                    for a in a_all])
    Mb_all = array([[[b[0], -b[1], -b[2], -b[3]],
                     [b[1],  b[0], -b[3],  b[2]],
                     [b[2],  b[3],  b[0], -b[1]],
                     [b[3], -b[2],  b[1],  b[0]]]
                    for b in b_all])

    M = tensordot(Ma_all, Mb_all, axes=([0,2],[0,2]))

    E, D = np.linalg.eig(M)

    max_idx = E.argmax()

    e = D[:,max_idx]

    M1 = array([[e[0], -e[1], -e[2], -e[3]],
                [e[1],  e[0],  e[3], -e[2]],
                [e[2], -e[3],  e[0],  e[1]],
                [e[3],  e[2], -e[1],  e[0]]])

    M2 = array([[e[0], -e[1], -e[2], -e[3]],
                [e[1],  e[0], -e[3],  e[2]],
                [e[2],  e[3],  e[0], -e[1]],
                [e[3], -e[2],  e[1],  e[0]]])

    R4 = dot(M1.T, M2)

    R = R4[1:,1:]

    T = centroid_world - dot(R,centroid_body)

    # The root sum of squared error
    err = sqrt(sum((dot(R,body) + T[:,newaxis] - world)**2))

    return R, T, err

def rigit_nn(body, world):
    p = shape(body)[1]; q = shape(world)[1]

    idx_corresp = zeros(p, dtype=int)
    for i in range(p):
        dist_list = ((body[:,i][:,newaxis] - world)**2).sum(0)
        idx_corresp[i] = dist_list.argmin()

    return rigit2(body, world[:,idx_corresp]), idx_corresp

def rigit_ransac(body, world, max_iters, tol, hint=None):
    assert hasattr(itertools, 'permutations'), \
        'Needs itertools.permutations(). Please make sure the system runs Python 2.6 or higher.'
    assert shape(world)[0] == 2 and shape(body)[0] == 2, \
        'Only accepts points in R^2. Maybe try transposing the data matrices?'

    p = shape(body)[1]; q = shape(world)[1]
    assert p > 0 and q > 0, \
        "Insufficiently many world or reference body points."
    max_slength = min(min(p,q), 4)
    if hint is not None:
        hint = [hi for hi in hint if hi < q]
        if len(hint) < max_slength:
            hint = None
    num_iter = 0
    err = 1e40                  # some large number
    R, T = None, None

    # Support occlusion of up to one point
    max_iters /= 2
    for slength in (max_slength, max_slength-1):
        while num_iter < max_iters and err > tol:
            if hint is not None:
                world_rand_idx = hint
                hint = None
            else:
                world_rand_idx = np.random.permutation(q)
            world_rand_pts = world[:, world_rand_idx[:slength]]

            for body_rand_idx in itertools.permutations(range(p), slength):
                body_rand_pts = body[:, body_rand_idx]
                R, T, err = rigit2(body_rand_pts, world_rand_pts)

                if err <= tol:
                    
                    # world_pts_tr = dot(R.T, world - T[:,newaxis])
                    # (local_R, local_T, local_err), idx_corresp = rigit_nn(body, world_pts_tr)
                    # print "\t\t", local_err
                    # R = np.dot(local_R, R)
                    # T = np.dot(local_R, T) + local_T
                    break

            num_iter = num_iter + 1

        if err <= tol:
            is_successful = True
            break
        else:
            is_successful = False
        
    return R, T, err, world_rand_idx[:slength], is_successful, num_iter

if __name__ == '__main__':

    body = np.random.rand(2,10)
    
    theta = 0.0
    R = array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]])
    T = array([1., 2.])

    world = dot(R, body) + T[:,newaxis]

    #  print body
    # print world

    # time_cur = time.time()
    # for i in range(1):

    #     R_est, T_est, err = rigit(body, world)

    # print 'Time elapsed = %g (s)' % (time.time() - time_cur)

    # print R_est
    # print T_est
    # print err

    time_start = time.time()
    R_ransac, T_ransac, err, idx_corresp, is_successful, num_iter \
        = rigit_ransac(body, world[:,:8], 10, 1e-8)
    time_elapsed = time.time() - time_start
    print 'total iterations = %d' % (num_iter)
    print 'Time elapsed = %g (s)' % (time_elapsed)
    print 'Iterations per second = %g' % (num_iter/time_elapsed)
    print R
    print R_ransac
    print T_ransac
    print err
    print idx_corresp
