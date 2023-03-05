#!/usr/bin/env python
#
# Author:  Steven C. Howell and Xiangyun Qiu
# Purpose: Basic geometric operations
# Created: 23 February 2015
#
# 0000000011111111112222222222333333333344444444445555555555666666666677777777778
# 2345678901234567890123456789012345678901234567890123456789012345678901234567890
#%%
# import sys
# import os
# import os.path as op
# import argparse #http://docs.python.org/dev/library/argparse.html
import numpy as np
import logging
from scipy.optimize import curve_fit
# import sasmol.sasmol as sasmol

# import logging
# LOGGER = logging.getLogger(__name__) #add module name manually

class MainError(Exception):
    pass

class struct():
    def __init__(self):
        pass

def align_kabsch_get(xyz_moving, xyz_fixed, scaling=False) :
    """ Adopted from internet resource kabsch.py """ 
    assert len(xyz_moving) == len(xyz_fixed), \
            'Error: moving and fixed xyz must have the same dimensions!!!'
    
    num_points = xyz_moving.shape[0]
    center_moving = np.mean(xyz_moving, axis=0)
    center_fixed = np.mean(xyz_fixed, axis=0)

    xyz_moving_centered = xyz_moving - np.tile(center_moving, (num_points,1))
    xyz_fixed_centered = xyz_fixed - np.tile(center_fixed, (num_points,1))

    # H = np.transpose(xyz_moving_centered)*xyz_fixed_centered
    H = np.dot(np.transpose(xyz_moving_centered), xyz_fixed_centered)

    if scaling :
        H = H/num_points

    U, S, Vt = np.linalg.svd(H)

    # rota_mat = Vt.T * U.T
    rota_mat = np.dot(Vt.T, U.T)

    if np.linalg.det(rota_mat) < 0 :
        logging.warning('Reflection detected')
        print('Warning: reflection detected!!!')
        Vt[2, :] *= -1
        rota_mat = np.dot(Vt.T, U.T)
        # rota_mat = Vt.T * U.T

    if scaling :
        var_fixed = np.var(xyz_fixed, axis=0).sum()
        scale_factor = var_fixed / np.sum(S)
        tran_mat = np.dot(-rota_mat, (center_moving.T * scale_factor)) + center_fixed.T
    else :
        scale_factor = 1
        tran_mat = np.dot(-rota_mat, center_moving.T) +  center_fixed.T

    return scale_factor*rota_mat, tran_mat # could return scale_factor separately

def align_kabsch_apply(xyz_moving, scale=1.0, rotate=None, translate=None) :
    num_points = xyz_moving.shape[0]
    xyz_moved = scale*np.dot(rotate, xyz_moving.T) + \
        np.tile(translate.reshape(translate.size,1), (1, num_points))
    return xyz_moved.T

def align_kabsch_test(scaling=False, scale=1.0) :
    # Test
    A = np.matrix([[10.0,10.0,10.0],
                [20.0,10.0,10.0],
                [20.0,10.0,15.0]]) * scale

    B = np.matrix([[18.8106,17.6222,12.8169],
                [28.6581,19.3591,12.8173],
                [28.9554, 17.6748, 17.5159]])

    n = B.shape[0]

    Ttarg = np.matrix([[0.9848, 0.1737,0.0000,-11.5859],
                    [-0.1632,0.9254,0.3420, -7.621],
                    [0.0594,-0.3369,0.9400,2.7755],
                    [0.0000, 0.0000,0.0000,1.0000]])

    Tstarg = np.matrix([[0.9848, 0.1737,0.0000,-11.5865],
                    [-0.1632,0.9254,0.3420, -7.621],
                    [0.0594,-0.3369,0.9400,2.7752],
                    [0.0000, 0.0000,0.0000,1.0000]])

    # recover the transformation
    ret_R, ret_t = align_kabsch_get(B, A, scaling=scaling)
    #s, ret_R, ret_t = umeyama(A, B)

    # Find the error
    B2 = ret_R * B.T + np.tile(ret_t, (1, n))
    B2 = B2.T
    err = A - B2
    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = np.sqrt(err / n);

    #convert to 4x4 transform
    match_target = np.zeros((4,4))
    match_target[:3,:3] = ret_R
    match_target[0,3] = ret_t[0]/scale
    match_target[1,3] = ret_t[1]/scale
    match_target[2,3] = ret_t[2]/scale
    match_target[3,3] = 1

    print("Points fixed")
    print(A)
    print("")

    print("Points moving")
    print(B)
    print("")

    print("Rotation")
    print(ret_R)
    print("")

    print("Translation")
    print(ret_t)
    print("")

    # print("Scale")
    # print(s)
    # print("")

    print("Homogeneous Transform")
    print(match_target)
    print("")

    if scaling:
        print("Total Diff to SA matrix")
        print((np.sum(match_target - Tstarg)))
        print("")
    else:
        print("Total Diff to SA matrix")
        print((np.sum(match_target - Ttarg)))
        print("")

    print(("RMSE:", rmse))
    print("If RMSE is near zero, the function is correct!")

def get_move_to_origin_matrix(coor4):
    '''
    construct the 4x4 matrix to move Nx4 coordinate to the origin (the first point)
    '''

    translation_matrix = np.eye(4)
    reverse_translation_matrix = np.copy(translation_matrix)

    try:
        translation_matrix[3, 0:3] = -coor4[0, 0:3]
        reverse_translation_matrix[3, 0:3] = coor4[0, 0:3]
    except:
        print('no coordinates to translate:\n', coor4)

    return translation_matrix, reverse_translation_matrix


def align_v_to_z(v):
    '''
    align the axis connecting the first 2 coordinates of a 1x4 array of
    coodinate vectors to the z-axis
    '''
    align = np.eye(4, dtype=np.float)
    v1 = np.array([0, 0, 1])
    v2 = v
    align[:3, :3] = rotate_v2_to_v1(v1, v2).transpose()

    return align


def get_alignment_angles(axes1, axes2):
    '''
    determine the angles to align axes2 to axes1

    Parameters
    ----------
    axes1 : 3x3 np.array
        The axes to rotate to
    axes2 : 3x3 np.array
        The axes to rotate from


    Returns
    -------
    phi : 1x3 np.array
        phi_x, phi_y, and phi_z angles to rotate axes2 about
    Notes
    -----
    to align axes, rotation order should be:
        phi_x about axes1[0]
        phi_y about axes1[1]
        phi_z about axes1[2]

    the reverse alignment should be
        -phi_z about axes1[2]
        -phi_y about axes1[1]
        -phi_x about axes1[0]


    procedure:
        1) reorient axes1 and axes2 so axes1 is along the cardinal directions
        2) determine phi_x
        3) determine phi_y
        4) determine phi_z

    See Also
    --------
    align_to_z : aligns the axis connecting the first 2 coordinates of a 1x4
                 array of coordinate vectors to be on the z-axis
    rotate_about_v : rotate coordinates about an arbitrary vector

    Examples
    --------
    >>> axes1 = np.eye(3)
    >>> sqrt2 = np.sqrt(2)
    >>> axes2 = np.array([[0, 0, 1],[1/sqrt2, 1/sqrt2, 0],[-1/sqrt2, 1/sqrt2, 0]])
    >>> phi = get_alignment_angles(axes1, axes2)
    >>> print phi
    [ 90.  45.  90.]
    '''

    # 0) make sure input axes are orthonomal
    assert np.allclose(
        np.eye(3), axes1.dot(axes1.transpose())), 'ERROR: axes1 is not orthonormal'
    assert np.allclose(
        np.eye(3), axes2.dot(axes2.transpose())), 'ERROR: axes2 is not orthonormal'

    # 1) reorient axes1 and axes2 so axes1 is along the cardinal directions
    if np.allclose(axes1, np.eye(3)):
        axes1xyz = axes1
        axes2xyz = axes2
    else:
        alignX = rotate_v2_to_v1(np.array([1, 0, 0]), axes1[0]).transpose()
        axes1x = axes1.dot(alignX)
        alignY = rotate_v2_to_v1(np.array([0, 1, 0]), axes1x[1]).transpose()
        axes1xy = axes1x.dot(alignY)
        alignZ = rotate_v2_to_v1(np.array([0, 0, 1]), axes1xy[2]).transpose()
        axes1xyz = axes1xy.dot(alignZ)
        axes2xyz = axes2.dot(alignX).dot(alignY).dot(alignZ)
        assert np.allclose(
            np.eye(3), axes1xyz), 'ERROR: failed to align axes1 to the cardinal axes'

    # 2) determine phi_x
    # angle to make z[1] = 0
    phi_x_r = np.arctan2(axes2xyz[2, 1], axes2xyz[2, 2])
    phi_x = phi_x_r * 180 / np.pi
    axes2xyz = rotate_about_v(
        np.concatenate(([[0, 0, 0]], axes2xyz)), axes1xyz[0], phi_x)[1:]

    # 3) determine phi_y
    # angle to make z[0] = 0
    phi_y_r = np.arctan2(-axes2xyz[2, 0], axes2xyz[2, 2])
    phi_y = phi_y_r * 180 / np.pi
    axes2xyz = rotate_about_v(
        np.concatenate(([[0, 0, 0]], axes2xyz)), axes1xyz[1], phi_y)[1:]

    # 4) determine phi_z
    # angle to make x[1] = 0 (y[0] = 0 also)
    phi_z_r = np.arctan2(-axes2xyz[0, 1], axes2xyz[0, 0])
    phi_z = phi_z_r * 180 / np.pi
    axes2xyz = rotate_about_v(
        np.concatenate(([[0, 0, 0]], axes2xyz)), axes1xyz[2], phi_z)[1:]

    # 5) verify the result actually rotated axes2 to axes1
    assert np.allclose(
        np.eye(3), axes2xyz), 'ERROR: failed to align axes2 to axes1 (invalid angles)'

    return np.array([phi_x, phi_y, phi_z])


def rotate_about_v(coor3, v, theta):
    '''
    this function is designed to generate a modified version of the input
    coordinates (coor3)
    1) translate all coordinates so the first one is at the origin
       (rotation origin should be first coordinate)
    2) orient the rotation vector, v, along the z-axis
    3) performs rotations using the angle theta
    4) reverse the orientation of v to the z-axis
    5) reverse transaltion of all coordinates so the first is where it started
    '''
    n_atoms = coor3.shape[0]

    # populate the arrays with the input values
    # changing coordinate array from 3 to 4 component vectors
    # to incorporate transaltions into the matrix math
    coor4 = np.ones((n_atoms, 4), np.float)
    coor4[:, 0:3] = coor3

    # create the translation-rotation matrix
    # This is intended to be multiplied from the right (unlike standard matrix
    # multiplication) so as not to require transposing the coordinate vectors.
    theta_xyz_rad = np.array([0, 0, theta]) * np.pi / 180.0  # radians

    [cx, cy, cz] = np.cos(theta_xyz_rad)
    [sx, sy, sz] = np.sin(theta_xyz_rad)

    # initialize the rotation
    # consolidated method of defining the rotation matrices
    rotate = np.eye(4, dtype=np.float)
    rotate[0][0:3] = [cy * cz,          cy * sz,          -sy]
    rotate[1][0:3] = [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy]
    rotate[2][0:3] = [sx * sz + cx * sy * cz, cx * sy * sz - sx * cz, cx * cy]

    # 1) move coordinates to rotation origin
    move_to_origin, return_from_origin = get_move_to_origin_matrix(coor4)
    # note: Ti0 != T0, negative off diag elements
    coor4 = np.dot(coor4, move_to_origin)

    # 2-3) align coordinates to the rotation vector then rotate
    z_align_matrix = align_v_to_z(v)
    align_and_rotate = np.dot(z_align_matrix, rotate)
    coor4 = np.dot(coor4, align_and_rotate)

    # 4) reverse alignment
    coor4 = np.dot(coor4, z_align_matrix.transpose())

    # 5) return rotation origin to the original position
    coor4 = np.dot(coor4, return_from_origin)

    # return the modified positions
    return coor4[:, 0:3]


def angle_btwn_v1_v2(v1, v2):
    '''
    get the angle between two arbitrary vectors

    Parameters
    ----------
    v1 : Nx3 np.array
        set 1 of the vector/s to use in the calculation
    v2 : Nx3 np.array
        set 2 of the vector/s to use in the calculation

    Returns
    -------
    theta_deg : Nx1 np.array
        The angles between v1 and v2 in degrees
    theta_rad : Nx1 np.array
        The angles between v1 and v2 in radians

    Notes
    -----
    Simple implementation of the cosine formula: cos \theta = u dot v

    See Also
    --------
    rorate_v2_to_v1: get the rotation matrix using this angle

    Examples
    --------
    >>> v1 = np.array([0,0,1])
    >>> v2 = np.array([1,0,0])
    >>> theta_deg, theta_rad = angle_btwn_v1_v2(v1, v2)
    >>> print theta_deg
    90
    >>> np.dot(R, v2) - v1
    array([ 0.,  0.,  0.])

    '''
    if len(v2.shape) != len(v1.shape):
        # pad the arrays to have the same length
        if len(v2.shape) > len(v1.shape):
            assert len(v1) == 1, ('ERROR: vector lengths are not compatible, '
                                  'review input')
            new_v1 = np.zeros(v2.shape)
            new_v1[:] = v1
            v1 = new_v1
        else:
            assert len(v2) == 1, ('ERROR: vector lengths are not compatible, '
                                  'review input')
            new_v2 = np.zeros(v1.shape)
            new_v2[:] = v2
            v2 = vew_v2

    if len(v2.shape) == 1:
        # make sure v1 and v2 are unit vectors:
        v1 = v1 / np.sqrt(v1.dot(v1))
        v2 = v2 / np.sqrt(v2.dot(v2))

        c = np.dot(v2, v1)

    else:
        # make sure v1 and v2 are unit vectors:
        v1 = np.einsum(
            'ij,i->ij', v1, 1 / np.sqrt(np.einsum('ij,ij->i', v1, v1)))
        v2 = np.einsum(
            'ij,i->ij', v2, 1 / np.sqrt(np.einsum('ij,ij->i', v2, v2)))

        # this could fail if a v1, v2 pair are nearly equal -> s~=0
        c = np.einsum('ij,ij->i', v2, v1)

    theta_rad = np.arccos(c)
    theta_deg = theta_rad / np.pi * 180.0
    return theta_deg, theta_rad


def dihedral_btwn_v1_v2(v1, v2, x1, x2):
    '''
    get the dihedral angle between two vectors defined by four points ()

    Parameters
    ----------
    v1 : Nx3 np.array
        set 1 of the vector/s to use in the calculation
    v2 : Nx3 np.array
        set 2 of the vector/s to use in the calculation
    x1 : Nx3 np.array
        set 1 of the origin/s to use in the calculation
    x2 : Nx3 np.array
        set 2 of the origin/s to use in the calculation

    Returns
    -------
    theta_deg : Nx1 np.array
        The angles between v1 and v2 in degrees
    theta_rad : Nx1 np.array
        The angles between v1 and v2 in radians


    Notes
    -----
    Implementated from: http://math.stackexchange.com/a/47084/221321
    According to the definition on wikipedia
    (https://en.wikipedia.org/wiki/Dihedral_angle),
    the direction of the angle does not use the right-hand-rule along
    b1.  It is defined using the negative or the left-hand-rule.

    See Also
    --------
    rorate_v2_to_v1: get the rotation matrix using the angle between two vectors
    angle_btwn_v1_v2: get the angle between two vectors

    Examples
    --------
    >>> v1 = np.array([1, 1, 0])
    >>> v2 = np.array([-1, 0, 1])
    >>> x1 = np.array([1, 0, 0])
    >>> x2 = np.array([-1, 0, 0])
    >>> theta_deg, theta_rad = geometry.dihedral_btwn_v1_v2(v1, v2, x1, x2)
    >>> print theta_deg
    90.0
    >>> print theta_rad
    1.57079632679
    >>> print theta_rad-np.pi/2
    0.0
    >>> v2 = np.array([-1, -1, 1])
    >>> theta_deg, theta_rad = geometry.dihedral_btwn_v1_v2(v1, v2, x1, x2)
    >>> print theta_deg
    135.0
    >>> print theta_rad-3*np.pi/4
    0.0

    '''

    assert v1.shape == v2.shape == x1.shape == x1.shape, (
        'ERROR: array sizes do not match, review input')

    if len(v1.shape) == 1:
        # vector between origins
        v3 = x2 - x1

        # make sure v1, v2, and v3 are unit vectors:
        b1 = -v1 / np.sqrt(v1.dot(v1)) # negative b/c want to point at x1
        b3 = v2 / np.sqrt(v2.dot(v2))
        b2 = v3 / np.sqrt(v3.dot(v3))

        # get the normal to the two planes:
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # get the angle between n1 and n2
        m1 = np.cross(n1, b2)
        x = n2.dot(n1)
        y = n2.dot(m1)
        theta_rad = np.arctan2(y, x)
        theta_deg = theta_rad / np.pi * 180.0

    else:
        # have not figured out cross products using einsum

        # setup the output
        theta_rad = np.zeros((len(v1),1))
        theta_deg = np.zeros((len(v1),1))

        # iteratively call self
        for i in range(len(v1)):
            theta_deg[i], theta_rad[i] = dihedral_btwn_v1_v2(v1[i], v2[i],
                                                             x1[i], x2[i])

    return theta_deg, theta_rad


def rotate_v2_to_v1(v1, v2):
    '''
    get the rotation matrix that rotate from v2 to v1 along the vector
    orthongal to both

    Parameters
    ----------
    v1 : Nx3 np.array
        The vector/s to rotate to
    v2 : Nx3 np.array
        The vector/s to rotate from


    Returns
    -------
    R : Nx3x3 np.array
        The rotation matrices that rotate from v2 to v1


    Notes
    -----
    Followed the description on http://math.stackexchange.com/questions/180418/

    See Also
    --------
    align_to_z : aligns the axis connecting the first 2 coordinates of a 1x4
                 array of coordinate vectors to be on the z-axis

    Examples
    --------
    >>> v1 = np.array([0,0,1])
    >>> v2 = np.array([1,0,0])
    >>> R = rotate_v2_to_v1(v1, v2)
    >>> print R
    [[ 0.  0. -1.]
     [ 0.  1.  0.]
     [ 1.  0.  0.]]
    >>> np.dot(R, v2) - v1
    array([ 0.,  0.,  0.])

    Graphical illustration:

    >>> NotImplemented
    NotImplemented
    '''
    if len(v2.shape) != len(v1.shape):
        if len(v2.shape) > len(v1.shape):
            new_v1 = np.zeros(v2.shape)
            new_v1[:] = v1
            v1 = new_v1
        else:
            pass  # THIS WILL BREAK

    if len(v2.shape) == 1:
        # make sure v1 and v2 are unit vectors:
        v1 = v1 / np.sqrt(v1.dot(v1))
        v2 = v2 / np.sqrt(v2.dot(v2))

        if np.allclose(v1, v2):
            R = np.eye(3)
        else:
            v = np.cross(v2, v1)  # hinge axis
            c = np.dot(v2, v1)
            s = np.sqrt(np.dot(v, v))
            V = np.array([[0, -v[2],  v[1]],
                          [v[2],     0, -v[0]],
                          [-v[1],  v[0],    0]])

            R = np.eye(3) + V + np.dot(V, V) * (1 - c) / s**2
    else:
        # make sure v1 and v2 are unit vectors:
        v1 = np.einsum(
            'ij,i->ij', v1, 1 / np.sqrt(np.einsum('ij,ij->i', v1, v1)))
        v2 = np.einsum(
            'ij,i->ij', v2, 1 / np.sqrt(np.einsum('ij,ij->i', v2, v2)))

        # this could fail if a v1, v2 pair are nearly equal -> s~=0
        v = np.cross(v2, v1)
        c = np.einsum('ij,ij->i', v2, v1)
        s = np.sqrt(np.einsum('ij,ij->i', v, v))

        n_v = len(v)

        # V = np.array([[    0, -v[2],  v[1]],
        # [ v[2],     0, -v[0]],
        # [-v[1],  v[0],    0]])
        V = np.zeros((n_v, 3, 3))
        V[:, 0, 1:] = v[:, 2:0:-1]
        V[:, 1, ::2] = v[:, ::-2]
        V[:, 2, :2] = v[:, 1::-1]
        V[:, 0, 1] = -V[:, 0, 1]
        V[:, 1, 2] = -V[:, 1, 2]
        V[:, 2, 0] = -V[:, 2, 0]

        I = np.zeros((n_v, 3, 3))
        I[:] = np.eye(3)

        R = I + V + \
            np.einsum(
                'ijk,i->ijk', np.einsum('hij,hjk->hik', V, V), (1 - c) / s**2)

    return R


def cylinder_distances_from_R(coor, R, X0, Y0, Vx, Vy):
    origin = np.array([X0, Y0, 0])  # Z0 = 0

    Vz = 1
    length = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    Ux = Vx / length
    Uy = Vy / length
    Uz = Vz / length
    U = np.array([Ux, Uy, Uz])

    coor_cyl = coor - origin
    # component of array from origin to point along axis
    A = np.dot(coor_cyl, U)
    D = coor_cyl - np.outer(A, U)               # vectors from axis to point
    dist = np.sqrt(np.square(D).sum(axis=1))  # distance from axis to point

    return dist - R


def vector_from_line(coor, origin, direction):
    [Vx, Vy, Vz] = direction
    length = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    Ux = Vx / length
    Uy = Vy / length
    Uz = Vz / length
    U = np.array([Ux, Uy, Uz])

    coor_cyl = coor - origin
    # component of array from origin to point along axis
    A = np.dot(coor_cyl, U)
    D = (coor_cyl) - np.outer(A, U)  # vectors from axis to point

    return D


def transform_coor(coor3, vector, origin):
    '''
    purpose:
        transform coordinates to a new cartesian configuration

    input parameters:
        coor3:  coordinates originating at [0, 0, 0] oriented along the
                z-axis [0, 0, 1]
        origin: final origin
        vector: final orientation

    output parameters:
        result: transformed coordinates

    improvements:
        rotate about the vector orthogonal to the two orientation vectors
        rather than about X, then Z axes
    '''
    # initialize vector arrays for coordinates and orientation vectors
    # changing them from 3 component vectors into 4 component vectors to
    # incorporate transaltions into the matrix math
    try:
        r, c = coor3.shape
        if c != 3:
            if r == 3:
                coor3 = coor3.T
    except:
        coor3 = coor3.reshape(1, 3)

    coor4 = np.ones((len(coor3), 4))
    coor4[:, 0:3] = coor3  # ; print 'coor4 =', coor4

    # angles to align original z-axis to vector

    # create the translation-rotation matrix
    # This is intended to be multiplied from the right (unlike standard matrix
    # multiplication) so as not to require transposing the coordinate vectors.
    vector = vector.reshape(3)
    [vx, vy, vz] = vector

    tz = np.arctan2(vx, vy)
    cz = np.cos(tz)
    sz = np.sin(tz)

    tx = np.arctan2(vx * sz + vy * cz, vz)

    # initialize the rotation about X, then Z:
    Rx = rotate('x', -tx)
    Rz = rotate('z', -tz)
    Rxz = np.dot(Rx, Rz)
    R = np.eye(4)
    R[:3, :3] = Rxz
    R[3, :3] = origin

    # perform the rotation
    result = np.dot(coor4, R)

    return result[:, :-1]

def rotate(axis, theta):
    R = np.eye(3)
    ct = np.cos(theta)
    st = np.sin(theta)
    if axis.lower() == 'x':
        (R[1, 1], R[1, 2]) = (ct, st)
        (R[2, 1], R[2, 2]) = (-st, ct)
    elif axis.lower() == 'y':
        (R[0, 0], R[0, 2]) = (ct, -st)
        (R[2, 0], R[2, 2]) = (st, ct)
    elif axis.lower() == 'z':
        (R[0, 0], R[0, 1]) = (ct, st)
        (R[1, 0], R[1, 1]) = (-st, ct)
    else:
        assert True, "ERROR!!! invalid rotation axis: {}".format(axis)
    return R

def transform_surf(X, Y, Z, vector, origin):
    r, c = X.shape

    coor = np.array(
        [X.reshape((r * c)), Y.reshape((r * c)), Z.reshape((r * c))]).T

    coor = transform_coor(coor, vector, origin)

    X_new = coor[:, 0].reshape((r, c))
    Y_new = coor[:, 1].reshape((r, c))
    Z_new = coor[:, 2].reshape((r, c))

    return X_new, Y_new, Z_new


def show_cylinder(coor, params, nuc_origin, nuc_axes, dyad_origin, dyad_axes):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    [R, X0, Y0, Vx, Vy] = params
    Z0 = 0
    Vz = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_cylinder_axis = False
    if plot_cylinder_axis:
        array = np.linspace(-24, 25)
        xs = array * Vx + X0
        ys = array * Vy + Y0
        zs = array * Vz + Z0
        ax.plot(xs, ys, zs, label='cylinder axis')

    plot_cylinder_origin = False
    if plot_cylinder_origin:
        # ax.scatter(X0, Y0, Z0, color='blue', label='origin', marker='x')
        cyl_origin = np.array([[X0, Y0, Z0]])
        # ax.plot(cyl_origin[:,0], cyl_origin[:,1], cyl_origin[:,2], color='blue', label='cylinder_origin', marker='x', line='')
        ax.plot(cyl_origin[:, 0], cyl_origin[:, 1],
                cyl_origin[:, 2], 'bx', label='cylinder origin')

    # plot the coordinates the cylinder was fit to
    ax.plot(coor[:, 0], coor[:, 1], coor[:, 2], 'o', label="C1' coordinates")

    # X,Y,Z = cylinder(np.ones((10,1))*R, 20)
    vector = np.array([Vx, Vy, Vz])
    X_raw, Y_raw, Z_raw = cylinder(np.ones((2, 1)) * R, 20)
    h = 40
    Z_raw = (Z_raw - 0.5) * h
    X, Y, Z = transform_surf(X_raw, Y_raw, Z_raw, vector, nuc_origin)
    ax.plot_wireframe(X, Y, Z, label='cylinder', color='orange', lw='2')

    ax.plot([nuc_origin[0]], [nuc_origin[1]], [
            nuc_origin[2]], 'gs', label='nucleosome origin')
    styles = ['r-', 'g-', 'b-']
    labels = ['X-axis', 'Y-axis', 'Z-axis']
    for (i, axis) in enumerate(nuc_axes):
        axes_vec = np.vstack((nuc_origin, axis * 15 + nuc_origin))
        ax.plot(axes_vec[:, 0], axes_vec[:, 1], axes_vec[
                :, 2], styles[i], label=labels[i])

    dyad_origin = dyad_origin.reshape(1, 3)
    ax.plot(dyad_origin[:, 0], dyad_origin[:, 1],
            dyad_origin[:, 2], 'rs', label='dyad origin')

    styles = ['r-', 'g-', 'b-']
    labels = ['dyad X-axis', 'dyad Y-axis', 'dyad Z-axis']
    for (i, axis) in enumerate(dyad_axes):
        dyad_axis = np.vstack((dyad_origin, axis * 10 + dyad_origin))
        ax.plot(dyad_axis[:, 0], dyad_axis[:, 1],
                dyad_axis[:, 2], styles[i], label=labels[i])
    # dyad = dyad_mol.coor()[0]
    # ax.plot(dyad[:,0], dyad[:,1], dyad[:,2], 'ro', label='dyad bp')

    # ax.plot_wireframe(X_raw,Y_raw,Z_raw, label='cylinder', color='red')
    # ax.scatter(0, 0, 0, color='red', label='data points', marker='x')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('R=%0.1f, X0=%0.1e, Y0=%0.1e, Vx=%0.1e, Vy=%0.1e' %
              (R, X0, Y0, Vx, Vy))
    plt.axis('equal')
    plt.legend(loc='upper left', numpoints=1)
    plt.show()


def show_ncp_geometry(all_ncp_plot_vars):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import x_dna.util.gw_plot as gwp

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    n_ncps = len(all_ncp_plot_vars)

    axis_mag = [90, 130]
    i = 0

    all_origins = [all_ncp_plot_vars[i].ncp_origin for i in range(n_ncps)]
    all_origins = np.array(all_origins)

    ncp_colors = [gwp.qual_color(0), gwp.qual_color(6),
                  gwp.qual_color(8), gwp.qual_color(2)]
    axes_colors = [gwp.qual_color(9), gwp.qual_color(1)]
    limits = []
    for i in range(n_ncps):
        ncp_origin = all_ncp_plot_vars[i].ncp_origin
        ncp_axes = all_ncp_plot_vars[i].ncp_axes
        # NCP axes
        labels = ['Cylinder Axes', 'Dyad Axes']
        for (j, axis) in enumerate(ncp_axes[[2, 0]]):
            axes_vec = np.vstack((axis * (-axis_mag[j] / 2) + ncp_origin,
                                  axis * (axis_mag[j] / 2) + ncp_origin))
            limits.append(axes_vec)
            if i == 0:
                ax.plot(axes_vec[:, 0], axes_vec[:, 1], axes_vec[:, 2],
                        c=axes_colors[j], label=labels[j], linewidth=3)
                ncp1_o = all_ncp_plot_vars[i].ncp_origin
                if j == 1:
                    ncp2_o = all_ncp_plot_vars[i + 2].ncp_origin
                    stack_axis = ncp2_o - ncp1_o
                    stack_axis = stack_axis / \
                        np.sqrt(stack_axis.dot(stack_axis))
                    stack_vec = np.vstack((stack_axis * (-axis_mag[1] / 2) + ncp1_o,
                                           stack_axis * (axis_mag[1] / 2) + ncp2_o))
                    limits.append(stack_vec)
                    ax.plot(stack_vec[:, 0], stack_vec[:, 1], stack_vec[:, 2],
                            c=gwp.qual_color(4), linewidth=3, label='Stack Axes')
                    ax.plot(all_origins[:, 0], all_origins[:, 1],
                            all_origins[:, 2], linewidth=2, c=gwp.qual_color(3),
                            label='Center-to-Center Segments')
            else:
                ax.plot(axes_vec[:, 0], axes_vec[:, 1], axes_vec[:, 2],
                        c=axes_colors[j], linewidth=3)

    center = np.array(all_origins).mean(axis=0)
    ax.plot([center[0]], [center[1]], [center[2]], marker='*',
            c=gwp.qual_color(7), ms=10, label='Array Center', linestyle='None')

    for i in range(n_ncps):
        coor = all_ncp_plot_vars[i].coor
        # plot the coordinates the cylinder was fit to
        coor_label = "NCP %d" % (i + 1)
        ax.plot(coor[:, 0], coor[:, 1], coor[:, 2], 'k', linewidth=5)
        ax.plot(coor[:, 0], coor[:, 1], coor[:, 2], c=ncp_colors[i], linewidth=4,
                label=coor_label)

    i = 1
    ncp1_o = all_ncp_plot_vars[i].ncp_origin
    ncp2_o = all_ncp_plot_vars[i + 2].ncp_origin
    stack_axis = ncp2_o - ncp1_o
    stack_axis = stack_axis / np.sqrt(stack_axis.dot(stack_axis))
    stack_vec = np.vstack((stack_axis * (-axis_mag[0] / 2) + ncp1_o,
                           stack_axis * (axis_mag[0] / 2) + ncp2_o))
    limits.append(stack_vec)
    ax.plot(stack_vec[:, 0], stack_vec[:, 1], stack_vec[:, 2],
            c=gwp.qual_color(4), linewidth=3)

    limits = np.array(limits).reshape(len(limits) * 2, 3)
    extent = (limits.max(axis=0) - limits.min(axis=0)).max() / 2
    ax.set_xlim([center[0] - extent, center[0] + extent])
    ax.set_ylim([center[1] - extent, center[1] + extent])
    ax.set_zlim([center[2] - extent, center[2] + extent])
    # DO NOT RESIZE THE IMAGE: it messes up the scale

    # Shrink current axis by 40% so the legend is visible
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend
    ax.set_axis_off()
    # plt.title('Nucleosome Array Geometry Definitions')

    lg = plt.legend(loc='upper left', numpoints=1, bbox_to_anchor=(1, 0.5))
    lg.draw_frame(False)
    plt.show()
    return


def draw_cube(r):
    # draw cube taken from:
    # https://github.com/matplotlib/matplotlib/issues/1077
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*list(zip(s, e)))


def axisEqual3D(ax):
    '''
    this did not work in it's current state 06/09/15
    '''
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                        for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def show_ncps(all_ncp_plot_vars, title='NCP array', save_name=[]):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n_ncps = len(all_ncp_plot_vars)

    for i in range(n_ncps):
        coor = all_ncp_plot_vars[i].coor
        params = all_ncp_plot_vars[i].opt_params
        ncp_origin = all_ncp_plot_vars[i].ncp_origin
        ncp_axes = all_ncp_plot_vars[i].ncp_axes
        dyad_origin = all_ncp_plot_vars[i].dyad_origin
        i_ncp = i + 1
        [R, X0, Y0, Vx, Vy] = params
        Z0 = 0
        Vz = 1

        # plot the coordinates the cylinder was fit to
        coor_label = "NCP-%d C1' atoms" % i_ncp
        ax.plot(coor[:, 0], coor[:, 1], coor[:, 2], 'o', label=coor_label)

        # # plot the cylinder fit of the coordinates
        # vector = np.array([Vx, Vy, Vz])
        # X_raw, Y_raw, Z_raw = cylinder(np.ones((2,1))*R, 20)
        # h = 40
        # Z_raw = (Z_raw-0.5) * h
        # X, Y, Z = transform_surf(X_raw, Y_raw, Z_raw, vector, ncp_origin)
        # ax.plot_wireframe(X,Y,Z, color='orange', lw='2')

        # plot NCP origin and axes
        # origin_label = "NCP-%d origin" % n_ncp
        ax.plot([ncp_origin[0]], [ncp_origin[1]], [ncp_origin[2]], 'ms')
        styles = ['r-', 'g-', 'b-']
        labels = ['X-axes', 'Y-axes', 'Z-axes']
        for (j, axis) in enumerate(ncp_axes):
            axes_vec = np.vstack((ncp_origin, axis * 80 + ncp_origin))
            if i_ncp == n_ncps:
                ax.plot(axes_vec[:, 0], axes_vec[:, 1], axes_vec[
                        :, 2], styles[j], label=labels[j])
            else:
                ax.plot(
                    axes_vec[:, 0], axes_vec[:, 1], axes_vec[:, 2], styles[j])

        # plot the NCP dyad origin
        dyad_origin = dyad_origin.reshape(1, 3)
        if i_ncp == n_ncps:
            ax.plot(dyad_origin[:, 0], dyad_origin[:, 1],
                    dyad_origin[:, 2], 'ms', label='dyad origin')
        else:
            ax.plot(
                dyad_origin[:, 0], dyad_origin[:, 1], dyad_origin[:, 2], 'ms')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.axis('equal')
    plt.legend(loc='upper left', numpoints=1, bbox_to_anchor=(1, 0.5))
    if save_name:
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        print('eog %s &' % save_name)
    else:
        plt.show()


def get_dna_bp_reference_frame(dna_ids, bp_mol, dna_id_type='segname'):
    '''
    The x-axis points in the direction of the major groove along what would
    be the pseudo-dyad axis of an ideal Watson-Crick base-pair, i.e. the
    perpendicular bisector of the C1'...C1' vector spanning the base-pair.
    The y-axis runs along the long axis of the idealized base-pair in the
    direction of the sequence strand, parallel with the C1'...C1' vector,
    and displaced so as to pass through the intersection on the
    (pseudo-dyad) x-axis of the vector connecting the pyrimidine Y(C6) and
    purine R(C8) atoms. The z-axis is defined by the right-handed rule,
    i.e. z = x cross y. (doi:10.1006/jmbi.2001.4987)
    '''
    c6c8_string = ('(((resname[i] == "CYT" or resname[i] == "THY") and name[i] == "C6") or'
                   ' ((resname[i] == "GUA" or resname[i] == "ADE") and name[i] == "C8"))')

    dna1_c1p_filter = '%s[i] == "%s" and name[i] == "C1\'" ' % (
        dna_id_type.lower(), dna_ids[0])
    dna2_c1p_filter = '%s[i] == "%s" and name[i] == "C1\'" ' % (
        dna_id_type.lower(), dna_ids[1])
    dna1_c6c8_filter = '%s[i] == "%s" and %s' % (
        dna_id_type.lower(), dna_ids[0], c6c8_string)
    dna2_c6c8_filter = '%s[i] == "%s" and %s' % (
        dna_id_type.lower(), dna_ids[1], c6c8_string)

    e0, dna1_c1p_mask = bp_mol.get_subset_mask(dna1_c1p_filter)
    e1, dna2_c1p_mask = bp_mol.get_subset_mask(dna2_c1p_filter)
    e2, dna1_c6c8_mask = bp_mol.get_subset_mask(dna1_c6c8_filter)
    e3, dna2_c6c8_mask = bp_mol.get_subset_mask(dna2_c6c8_filter)
    assert np.sum(dna1_c1p_mask) == np.sum(dna1_c6c8_mask) == np.sum(dna2_c1p_mask) == np.sum(
        dna2_c6c8_mask), "ERROR: input did not contain atoms necessary for determining orientation"

    dna1_c1p = np.dot(dna1_c1p_mask, bp_mol.coor()[0])
    dna2_c1p = np.dot(dna2_c1p_mask, bp_mol.coor()[0])
    dna1_c6c8 = np.dot(dna1_c6c8_mask, bp_mol.coor()[0])
    dna2_c6c8 = np.dot(dna2_c6c8_mask, bp_mol.coor()[0])

    y_vec = dna1_c1p - dna2_c1p
    y_mag = np.sqrt(np.dot(y_vec, y_vec))
    y_hat = y_vec / y_mag

    # following: http://geomalgorithms.com/a05-_intersect-1.html
    Q0 = (dna1_c1p + dna2_c1p) / 2
    P0 = dna2_c6c8
    P1 = dna1_c6c8
    w = P0 - Q0
    u = P1 - P0
    s1 = np.dot(-y_hat, w) / np.dot(y_hat, u)
    assert 0 <= s1 <= 1, "ERROR: problem in calculating bead origin"
    bp_origin = P0 + s1 * u

    a = bp_origin - dna2_c1p
    x_vec = a - np.dot(a, y_hat) * y_hat
    x_hat = x_vec / np.sqrt(np.dot(x_vec, x_vec))

    z_hat = np.cross(x_hat, y_hat)

    bp_axes = np.array([x_hat, y_hat, z_hat])

    return bp_origin, bp_axes


def cylinder(r, n=20):
    '''
    Source: http://python4econ.blogspot.com/2013/03/matlabs-cylinder-command-in-python.html
    Returns the unit cylinder that corresponds to the curve r.
    INPUTS:  r - a vector of radii
             n - number of coordinates to return for each element in r

    OUTPUTS: x,y,z - coordinates of points
    '''

    # ensure that r is a column vector
    r = np.atleast_2d(r)
    r_rows, r_cols = r.shape

    # added to make it so the result is not just a circle
    if r_rows == r_cols == 1:
        r = np.ones((2, 1)) * r

    if r_cols > r_rows:
        r = r.T

    # find points along x and y axes
    points = np.linspace(0, 2 * np.pi, n + 1)
    x = np.cos(points) * r
    y = np.sin(points) * r

    # find points along z axis
    rpoints = np.atleast_2d(np.linspace(0, 1, len(r)))
    z = np.ones((1, n + 1)) * rpoints.T

    return x, y, z


def get_dna_bp_and_axes(bp_mask, dna_ids, dna_mol, bp_mol=None, dna_id_type='segname'):
    if not bp_mol:
        bp_mol = sasmol.SasMol(0)
        error = dna_mol.copy_molecule_using_mask(bp_mol, bp_mask, 0)
    else:
        error, bp_coor = dna_mol.get_coor_using_mask(0, bp_mask)
        bp_mol.setCoor(bp_coor)
    bp_origin, bp_axes = get_dna_bp_reference_frame(
        dna_ids, bp_mol, dna_id_type)

    return bp_origin, bp_axes, bp_mol


def get_axes_from_points(origin, p1, p2):
    '''
    determine the orthogonal coordinate axes using an origin point and 2 points

    Parameters
    ----------
    origin : np.array
        The origin for the coordinate axes
    p1 : np.array
        A point along the first axis
    p2 : np.array
        A point along the second axis

    Returns
    -------
    ax1_hat : np.array
        unit vector along the first axis
    ax2_hat : np.array
        unit vector along the second axis
    ax3_hat : np.array
        unit vector along the third axis

    Notes
    -----
    The third axis will be determined using right-hand-rule:
    ax3_hat = ax1_hat x ax2_hat

    accordingly (p1, p2) should be (pX, pY), (pY, pZ), or (pZ, pX)

    example:
    >>> get_axes_from_points(np.array([0,0,0]), np.array([2,0,0]),
                             np.array([0,3,0]))
    (array([ 1.,  0.,  0.]), array([ 0.,  1.,  0.]), array([ 0.,  0.,  1.]))
    '''
    ax1 = p1 - origin
    ax2 = p2 - origin
    ax1_hat = ax1 / np.sqrt(np.dot(ax1, ax1))
    ax2_hat = ax2 / np.sqrt(np.dot(ax2, ax2))
    ax3_hat = np.cross(ax1_hat, ax2_hat)

    return ax1_hat, ax2_hat, ax3_hat


def get_ncp_origin_and_axes(ncp_c1p_mask, dyad_mask, dyad_dna_id, ncp,
                            ref_atom_mask, prev_opt_params=None,
                            dna_id_type='segname', dyad_mol=None, debug=False):
    dyad_origin, dyad_axes, dyad_mol = get_dna_bp_and_axes(dyad_mask,
                                                           dyad_dna_id, ncp, dyad_mol, dna_id_type)

    error, coor = ncp.get_coor_using_mask(0, ncp_c1p_mask)
    coor = coor[0]
    ideal = np.zeros(len(coor))

    # fit a cylinder
    if debug:
        import time
        tic = time.time()

    '''
    try using the previous optimization parameters
    this works well if it has not moved far, otherwise it can be bad
    '''
    # if prev_opt_params is None:
        # x = no_such_var
    # else:
        # opt_params, cov_params = curve_fit(cylinder_distances_from_R,
                                           # coor, ideal, p0=prev_opt_params)

    '''
    use the dyad_z_axis as the guess for the ncp_z_axis
    this consistently works well
    '''
    R = 41.5
    dyad_y_axis = dyad_axes[1, :] / dyad_axes[1, 2]

    # guess where the cylinder crosses the z=0 plane
    ncp_origin_guess = dyad_origin + R / 2 * dyad_axes[0]
    cyl_origin_guess = ncp_origin_guess - \
        ncp_origin_guess[2] / dyad_axes[1, 2] * dyad_axes[1]

    guess = np.array([R, cyl_origin_guess[0], cyl_origin_guess[1], dyad_y_axis[
                     0], dyad_y_axis[1]])  # (R, X0, Y0, Vx, Vy)
    opt_params, cov_params = curve_fit(cylinder_distances_from_R, coor,
                                       ideal, p0=guess)
    if debug:
        toc = time.time() - tic
        print('fitting a cylinder to the NCP took %0.3f s' % toc)

    [R, X0, Y0, Vx, Vy] = opt_params
    Z0 = 0
    Vz = 1
    cyl_origin = np.array([X0, Y0, Z0])
    z = np.array([Vx, Vy, Vz])
    z_hat = z / np.sqrt(np.dot(z, z))

    # calculate distance from dyad_orign to the axis
    x = vector_from_line(dyad_origin, np.concatenate(
        (opt_params[1:3], [0])), np.concatenate((opt_params[3:5], [1])))
    x = x.reshape(3)
    ncp_origin = dyad_origin - x
    x_hat = x / np.sqrt(np.dot(x, x))

    # xp0 = vector_from_cylinder_axis(dyad_origin, params[0], params[1], params[2], params[3], params[4])
    # xp1 = xp0-origin
    # x = xp1 - np.dot(xp1, z_hat)*z_hat #subtract from x the projection along
    # z_hat

    y_hat = np.cross(z_hat, x_hat)
    error, ref_coor = ncp.get_coor_using_mask(0, ref_atom_mask)
    ref_vec = (ref_coor - ncp_origin).reshape(y_hat.shape)
    ref_vec /= np.sqrt(ref_vec.dot(ref_vec))
    if y_hat.dot(ref_vec) < 0:
        y_hat *= -1
        z_hat *= -1
    ncp_axes = np.array([x_hat, y_hat, z_hat])

    if debug:
        # display the fit results
        show_cylinder(coor, opt_params, ncp_origin, ncp_axes, dyad_origin,
                      dyad_axes)

    ncp_plot_vars = struct()
    ncp_plot_vars.coor = coor
    ncp_plot_vars.opt_params = opt_params
    ncp_plot_vars.ncp_origin = ncp_origin
    ncp_plot_vars.ncp_axes = ncp_axes
    ncp_plot_vars.dyad_origin = dyad_origin
    ncp_plot_vars.dyad_axes = dyad_axes

    return ncp_origin, ncp_axes, opt_params, dyad_mol, ncp_plot_vars

if __name__ == 'TO BE IMPLEMENTED' : # '__main__':

    axes1 = np.eye(3)
    sqrt2 = np.sqrt(2)
    axes2 = np.array(
        [[0, 0, 1], [1 / sqrt2, 1 / sqrt2, 0], [-1 / sqrt2, 1 / sqrt2, 0]])
    phi = get_alignment_angles(axes1, axes2)
    print(phi)

    import time
    pdb_file = '1KX5tailfold_167bp.pdb'
    ncp = sasmol.SasMol(0)
    ncp.read_pdb(pdb_file)
    basis_filter = '( chain[i] ==  "I"  or chain[i] ==  "J"  ) and name[i] ==  "C1\'" '
    error, c1p_mask = ncp.get_subset_mask(basis_filter)
    dyad_dna_resids = [0, 0]
    dyad_dna_id = ['I', 'J']

    bp_filter = ('( chain[i] == "%s" and resid[i] == %d ) or ( chain[i] == "%s" '
                 'and resid[i] == %d )' % (dyad_dna_id[0], dyad_dna_resids[0],
                                           dyad_dna_id[1], dyad_dna_resids[1]))
    error, dyad_mask = ncp.get_subset_mask(bp_filter)

    ref_atom_basis = 'segname[i] == "I" and resid[i] == 18 and name[i] == "C1\'"'
    error, ref_atom_mask = ncp.get_subset_mask(ref_atom_basis)

    tic = time.time()
    ncp_origin, ncp_axes, opt_params, dyad_mol, plot_vars = get_ncp_origin_and_axes(
        c1p_mask, dyad_mask, dyad_dna_id, ncp, ref_atom_mask,
        prev_opt_params=None, dna_id_type='chain', dyad_mol=None, debug=False)
    toc = time.time() - tic
    print('determining the NCP origin and axes took %0.3f s' % toc)
    print('ncp_origin =', ncp_origin)
    print('ncp_axes =\n', ncp_axes)

    # coor = np.array([[0,0,1]])
    # vector = np.array([params[3], params[4], 1])
    # origin = np.array([params[2], params[3], 0])
    # # v = np.ones(3)
    # # origin = np.zeros(3)
    # res = transform_coor(coor, vector, origin)
    # print '|res|:', res/np.sqrt(np.dot(res,res))
    # print 'should parrallel'
    # print '|v|:', vector/np.sqrt(np.dot(vector,vector))

    # print('\m/ >.< \m/')
