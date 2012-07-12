# test_bezier.py --- 
# 
# Filename: test_bezier.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed Jul 11 11:21:50 2012 (+0530)
# Version: 
# Last-Updated: Thu Jul 12 19:10:45 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 412
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This provides utility for computing smooth arrows between three or
# more elements. Use the get_control_points_3 for three points. For
# more points use the get_control_points function.
# 
# 
# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

def bezier_angle(p0, p1):
    """Calculate the angle of the slope of cubic Bezier curve given
    two consecutive points. (i.e. (knot[0], cp[0]), or (cp[1],
    knot[1])).

    Example: angle with x-axis of the curve through knots [p0, p1, p2]
    and control points u0, v0 and u1, v1:

    at p0:
    bezier_angle(p0, u0)

    at p2:    
    bezier_angle(v1, p2)

    NOTE: this is not a general solution. It is for the end points of
    the curve.
    
    """
    if p1[0] == p0[0]:
        return np.pi/2
    return np.atan((p1[1] - p0[1])/(p1[0] - p0[0]))    

def bezier(p0, p1, p2, p3, t):
    """Evaluate cubic bezier curve going through p0, p1, p2 at
    parameter value t

    p0 is first end point, p3 is second end point, p1 first control
    point and p2 second control point.

    """
    p = np.vstack((p0, p1, p2, p3))
    s = 1 - t
    return s * (s * (s * p[0] + 3 * t * p[1]) + 3 * t * t * p[2]) + t * t * t * p[3]

def get_control_point_quad(p0, p1):
    """This is a convenience fuction to return the bounding retangle
    corner between points p0 and p1 as control point so that a nice
    curve can be drawn from p0 to p1 that is tangential to the axes at
    p0 and p1"""
    return (p1[0], p0[1])

def get_control_points_3(p1, p2, p3):
    """get the control points for ponts p1, p2 and p3"""
    p = np.vstack((p1, p2, p3))
    u = np.zeros((2,2))
    v = np.zeros((2,2))
    u[1] = p[1] + (p[2] - p[0])/6.0
    u[0] = (p[0] + 2*p[1] - u[1]) / 2.0
    v[0] = 2 * u[0] - p[0]
    v[1] = (p[2]+u[1])/2.0
    return (u, v)

def solve_tridiagonal(sub, dia, sup, b):
    """Solve tridiagonal system with sup[0:-1] containing supradiagonal, dia
    containing diagonal elements and sub[1:] containing subdiagonal
    elements."""
    m = sub[1:]/dia[:-1]
    dia[1:] -= m * sup[:-1]
    b[1:] -= m * b[:-1]
    x = np.zeros(b.shape)
    x[-1] = b[-1]/dia[-1]
    for i in range(len(b)-2, -1, -1):        
        x[i] = (b[i] - sup[i] * x[i+1])/dia[i]
        print x
    return x

def get_first_cp(knots):
    """Calculate the list of first control points for points specified
    in knots."""
    n = len(knots)-1
    sup = np.zeros((n, 2))
    sub = np.zeros((n,2))
    dia = np.zeros((n,2))
    b = np.zeros((n, 2))
    # 2 * u0 + u1 = p0 + 2 * p1
    b[0,:] = knots[0,:] + 2 * knots[1,:]
    # 3 * u[i] + 2 * u[i+1] + u[i+2] = 4 * p[i+1] + 2 * p[i+2]
    dia[:,:] = 2.0
    sup[:,:] = 1.0
    sub[:,:] = 3.0
    b[1:-1,:] = 4 * knots[1:-1,:] + 2 * knots[2:,:]
    # 2 * u[n-1] + 7 * u[n] = 8 * p[n-1] + p[n]
    dia[-1,:] = 7.0
    sub[-1,:] = 2.0    
    b[-1,:] = 8 * knots[n-1,:] + knots[n]
    return solve_tridiagonal(sub, dia, sup, b)

def get_second_cp(knots, first_cp):
    """Calculate the list of second control points for specified knots
    and first control points."""
    scp = np.zeros(first_cp.shape)
    scp[:-1] = 2 * knots[1:-1] - first_cp[1:]
    scp[-1] = (knots[-1] + first_cp[-1]) / 2.0
    return scp

def get_control_points(knots):
    """Get Bezier control points for a smooth Bezier curve passing through knots."""
    fcp = get_first_cp(knots)
    scp = get_second_cp(knots, fcp)
    return (fcp, scp)

def get_first_ctrl_points(rhs):
    """This is derived from:

    http://www.codeproject.com/Articles/31859/Draw-a-Smooth-Curve-through-a-Set-of-2D-Points-wit?fid=1532654&df=90&mpp=25&noise=3&prof=False&sort=Position&view=Quick&fr=1
    """
    tmp = np.zeros(rhs.shape)
    x = np.zeros(rhs.shape)
    b = np.ones(2) * 2.0
    x[0,:] = rhs[0,:] / b
    for ii in range(1, len(rhs)):
        tmp[ii,:] = 1/b
        if ii < len(rhs) - 1:
            b = 4.0 - tmp[ii]
        else:
            b = 3.5 - tmp[ii]
        x[ii] = (rhs[ii] - x[ii-1]) / b
    for ii in range(1, len(rhs)):
        x[len(rhs) - ii - 1][:] -= tmp[len(rhs) - ii] * x[len(rhs) - ii]
    return x
        
def get_curve_control_points(knots):
    """Returns array of first control points followed by that of
    second control points

    Parameters

    knots: 3x2 array containing the three poinst: [(x0, y0), (x1, y1),
    (x2, y2)] or more

    Returns
    
    (fcp, scp) where fcp is the first control point array and scp is
    second control point array. The control points for the curve
    between point[i] and point[i+1] are fcp[i] and scp[i].
    
    http://www.codeproject.com/Articles/31859/Draw-a-Smooth-Curve-through-a-Set-of-2D-Points-wit?fid=1532654&df=90&mpp=25&noise=3&prof=False&sort=Position&view=Quick&fr=1
    """
    if knots is None or len(knots) <= 2:
        raise ValueError('knots must be non-empty list of points with at least three entries')
    rhs = np.zeros((knots.shape[0] - 1, knots.shape[1]))
    rhs[0,:] = knots[0] + 2 * knots[1]
    rhs[1:-1,:] = 4 * knots[1:-2,:] + 2 * knots[2:-1,:]
    rhs[-1,:] = (8 * knots[-2,:]  + knots[-1,:]) / 2.0
    fcp = get_first_ctrl_points(rhs)
    scp = np.zeros(fcp.shape)
    scp[:-1,:] = 2 * knots[1:-1,:] - fcp[1:,:]
    scp[-1,:] = (knots[-1,:] + fcp[-1,:])/2.0
    return (fcp, scp)

if __name__ == '__main__':
    knots = np.array([[0, 1], [1, 1], [1, 0]])
    fcp, scp = get_curve_control_points(knots)
    print 'First control points:'
    print fcp
    print 'Second control points:'
    print scp
    print 'Mine\nfcp:\n'
    fcp1 = get_first_cp(knots)
    print fcp1
    print 'scp\n'
    print get_second_cp(knots, fcp1)
    print get_control_points_3(knots[0], knots[1], knots[2])

    # Display the example bezier curve through the knot points
    points = np.vstack([knots[0], fcp[0], scp[0], knots[1], knots[1], fcp[1], scp[1], knots[2]])
    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4]
    path = Path(points, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=2)    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_patch(patch)
    xs, ys = zip(*points)
    ax.plot(xs, ys, 'x--')
    p = bezier(knots[0], fcp[0], scp[0], knots[1], 0.5)
    plt.plot([p[0]], [p[1]], 'r^')
    plt.show()
    
# 
# test_bezier.py ends here
