# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:58:41 2024

@author: canu
3D alpha shape algorithm reference:Geun user in https://stackoverflow.com/
Takes a 3D point cloud (zeros array in the application) and an alpha parameter. Creates the triangulation of point cloud for the given alpha parameter. 
"""

from scipy.spatial import Delaunay
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra.vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra.vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices,Edges,Triangles
c010_3d = np.genfromtxt('c005.csv', delimiter=',', skip_header=1)
c010_3d_filter = c010_3d[c010_3d[:,-1]>10.5e-9]
c010_3d = c010_3d_filter

zeros = np.zeros((len(c010_3d),3))
zeros[:,0] = c010_3d[:,0] + c010_3d[:,3]
zeros[:,1] = c010_3d[:,1] + c010_3d[:,4]
zeros[:,2] = c010_3d[:,2] + c010_3d[:,5]
ones = c010_3d[c010_3d[:,2]==max(c010_3d[:,2])]
ones_2d = np.zeros((len(ones),2))
ones_2d[:,0] = ones[:,0] + ones[:,3]
ones_2d[:,1] = ones[:,1] + ones[:,4]
alpha=10
for i in range(10):

    verices, edges, triangles= alpha_shape_3D(zeros, alpha=alpha)
    fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    ugur = np.zeros((len(triangles), 3,3))
    for i in range(len(triangles)):
        ugur[i,:,:] = zeros[triangles[i,:],:]
        # mesh = Poly3DCollection(zeros[verices,:])
    mesh = Poly3DCollection(ugur)
    
    #
    
    ax = plt.axes(projection='3d')
    # ax.view_init(elev=30, azim=120)
    
    # ax.scatter(zeros[verices,0], zeros[verices,1], zeros[verices,2], s=200)
    ax.scatter(zeros[:,0], zeros[:,1], zeros[:,2], 'orange')
    mesh.set_edgecolor('r')
    
    ax.add_collection3d(mesh)
    # ax.view_init(elev=30, azim=0)
    
    ax.set_xlim(-0.05, 0.05)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(-0.05, 0.05)  # b = 10
    ax.set_zlim(-0.02, 0.02)  # c = 16
    alpha = np.round(alpha,5)
    
    plt.title('Alpha={0} Delaunay triangulation'.format(alpha))
    plt.tight_layout()
    plt.show()
    alpha = alpha / 2.5
