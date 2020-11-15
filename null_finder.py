#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
License:

Copyright (c) 2019 Federica Chiti, David Pontin, Roger Scott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

"""
Usage:

null_list = find_nulls(nx,ny,nz, xv,yv,zv, B_x1,B_y1,B_z1, tolerance=None)

This function checks the existence of null points and localize them in the grid

Parameters: nx, ny, nz, xv, yv, zv, B_x1, B_y1, B_z1, tolerence(=None)
nx, ny, nz represent the resolution of the grid
xv, yv, zv are the coordinate in the x, y and z directions
B_x1, B_y1, B_z1 are the components of the B field
tolerance sets the exit condition for Newton's method (root finding)

Returns: null_list
null_list is a concatenated list of coordinates for all detected nulls

"""


import numpy as np
import math
from numpy.linalg import inv

    
def find_nulls(nx,ny,nz,xv,yv,zv,B_x1,B_y1,B_z1, tolerance=None):
    null_list = []
    num_nulls = 0
    
    # calculating the sign change of the field components at the corners of the cells of the grid
    bx_sc = field_sign_change(B_x1)
    by_sc = field_sign_change(B_y1)
    bz_sc = field_sign_change(B_z1)
 
    # REDUCTION STAGE: keeping the indices of those cells for which the field components change sign at one of the vertices of the cells
    ind_list = np.array(np.where(bx_sc & by_sc & bz_sc)).T
   
    if not tolerance: tolerance = 10**-5

    # looping over the cells that pass the reduction stage
    for ind in ind_list:
        
        # retrieving the indices that satisfy the reduction stage
        i = ind[0]
        j = ind[1]
        k = ind[2]
 

         # trilinear interpolation
        
        tri_x = trilinear_coeffs(xv,yv,zv,i,j,k,B_x1)
        
        tri_y = trilinear_coeffs(xv,yv,zv,i,j,k,B_y1)
        
        tri_z = trilinear_coeffs(xv,yv,zv,i,j,k,B_z1)
       
        trilinear = np.array([tri_x,tri_y,tri_z])
        
        # BILINEAR STAGE
        # creating three lists that store the sign of each field component on the faces of the cube
        # the sign is appended only if the location given by the bilinear interpolation is on the face that is being considered
        
        bxx = []
        byy = []
        bzz = []
        
        # FACE 1
        # f is the parameter that tells the code if we're on an x/y/z face
        
        f = 0
        
        face1 = xv[i]
        
        # bx = 0 and by = 0
        # get bilinear coefficients
        bxby = x_face(face1,trilinear[0],trilinear[1])
        z_sign1 = bilinear(bxby,yv,zv,j,k,face1,tri_z,f)
        # append the sublist to the main list only if it is not empty
        if z_sign1:
            bzz.append(z_sign1)
        
        # by = 0 and bz = 0
        bybz = x_face(face1,trilinear[1],trilinear[2])
        x_sign1 = bilinear(bybz,yv,zv,j,k,face1,tri_x,f)
        if x_sign1:
            bxx.append(x_sign1)

        
        # bx = 0 and bz = 0
        bxbz = x_face(face1,trilinear[0],trilinear[2])
        y_sign1 = bilinear(bxbz,yv,zv,j,k,face1,tri_y,f)
        if y_sign1:
            byy.append(y_sign1)               
                
        
        # FACE 2
        
        f = 0
        
        face2 = xv[i+1]
        
        # bx = 0 and by = 0
        # get bilinear coefficients
        bxby = x_face(face2,trilinear[0],trilinear[1])
        z_sign2 = bilinear(bxby,yv,zv,j,k,face2,tri_z,f)
        if z_sign2:
            bzz.append(z_sign2)
        
        # by = 0 and bz = 0
        bybz = x_face(face2,trilinear[1],trilinear[2])
        x_sign2 = bilinear(bybz,yv,zv,j,k,face2,tri_x,f)
        if x_sign2:
            bxx.append(x_sign2)
        
        # bx = 0 and bz = 0
        bxbz = x_face(face2,trilinear[0],trilinear[2])
        y_sign2 = bilinear(bxbz,yv,zv,j,k,face2,tri_y,f)
        if y_sign2:
            byy.append(y_sign2)
                  
            
        # FACE 3
        
        f = 1
        
        face3 = yv[j]

        # bx = 0 and by = 0
        # get bilinear coefficients
        bxby = y_face(face3,trilinear[0],trilinear[1])
        z_sign3 = bilinear(bxby,xv,zv,i,k,face3,tri_z,f)
        if z_sign3:
            bzz.append(z_sign3)
            
        # by = 0 and bz = 0
        bybz = y_face(face3,trilinear[1],trilinear[2])
        x_sign3 = bilinear(bybz,xv,zv,i,k,face3,tri_x,f)     
        if x_sign3:
            bxx.append(x_sign3)
            
        # bx = 0 and bz = 0
        bxbz = y_face(face3,trilinear[0],trilinear[2])
        y_sign3 = bilinear(bxbz,xv,zv,i,k,face3,tri_y,f)      
        if y_sign3:
            byy.append(y_sign3)
  
        
        # FACE 4
        
        f = 1
        
        face4 = yv[j+1]

        
        # bx = 0 and by = 0
        # get bilinear coefficients
        bxby = y_face(face4,trilinear[0],trilinear[1])
        z_sign4 = bilinear(bxby,xv,zv,i,k,face4,tri_z,f)
        if z_sign4:
            bzz.append(z_sign4)            
            
        # by = 0 and bz = 0
        bybz = y_face(face4,trilinear[1],trilinear[2])
        x_sign4 = bilinear(bybz,xv,zv,i,k,face4,tri_x,f)       
        if x_sign4:
            bxx.append(x_sign4)
            
        # bx = 0 and bz = 0
        bxbz = y_face(face4,trilinear[0],trilinear[2])
        y_sign4 = bilinear(bxbz,xv,zv,i,k,face4,tri_y,f)
        if y_sign4:
            byy.append(y_sign4)
        
        
        # FACE 5
        
        f = 2
        
        face5 = zv[k]

        # bx = 0 and by = 0
        # get bilinear coefficients
        bxby = z_face(face5,trilinear[0],trilinear[1])
        z_sign5 = bilinear(bxby,xv,yv,i,j,face5,tri_z,f)
        if z_sign5:
            bzz.append(z_sign5)            
            
        # by = 0 and bz = 0
        bybz = z_face(face5,trilinear[1],trilinear[2])
        x_sign5 = bilinear(bybz,xv,yv,i,j,face5,tri_x,f)
        if x_sign5:
            bxx.append(x_sign5)
            
        # bx = 0 and bz = 0
        bxbz = z_face(face5,trilinear[0],trilinear[2])
        y_sign5 = bilinear(bxbz,xv,yv,i,j,face5,tri_y,f)
        if y_sign5:
            byy.append(y_sign5)
        
        
        # FACE 6
        
        f = 2
        
        face6 = zv[k+1]
        
        # bx = 0 and by = 0
        # get bilinear coefficients
        bxby = z_face(face6,trilinear[0],trilinear[1])
        z_sign6 = bilinear(bxby,xv,yv,i,j,face6,tri_z,f)        
        if z_sign6:
            bzz.append(z_sign6)
            
        # by = 0 and bz = 0
        bybz = z_face(face6,trilinear[1],trilinear[2])
        x_sign6 = bilinear(bybz,xv,yv,i,j,face6,tri_x,f)
        if x_sign6:
            bxx.append(x_sign6)
        
        # bx = 0 and bz = 0
        bxbz = z_face(face6,trilinear[0],trilinear[2])
        y_sign6 = bilinear(bxbz,xv,yv,i,j,face6,tri_y,f)
        if y_sign6:
            byy.append(y_sign6)
        
        # making flat lists
        bxx = [item for sublist in bxx for item in sublist]
        byy = [item for sublist in byy for item in sublist]
        bzz = [item for sublist in bzz for item in sublist]
        
        
        # if the function check_sign detects a change in sign in at least one of the three field components, then a single null point must exist in the cell
        # hence, apply Newton-Raphson method to find its location
        
        if (not check_sign(bxx)) or (not check_sign(byy)) or (not check_sign(bzz)):
        # if not (check_sign(bxx) and check_sign(byy) and check_sign(bzz)):

            # NEWTON RAPHSON METHOD
            
            # first guess: centre of the cube 
            
            xg = 0.5
            yg = 0.5
            zg = 0.5
            xs = xv[i]+(xv[i+1]-xv[i])*xg
            ys = yv[j]+(yv[j+1]-yv[j])*yg
            zs = zv[k]+(zv[k+1]-zv[k])*zg
            
            # grid size
            delta_x = xv[i+1]-xv[i]
            delta_y = yv[j+1]-yv[j]
            delta_z = zv[k+1]-zv[k]
            # values of solution
            x = [0]
            y = [0]
            z = [0]
            # step size
            step_x = []
            step_y = []
            step_z = []
            # error relative to the local grid size
            err_rel_grid = []
            # error relative to the solution
            err_rel_sol = []
            
            converged = False
       
            # set a counter to limit the number of iterations 
            n_steps = 0
            
            while (not converged) and (n_steps < 11): 
                n_steps += 1
                # calculating B field magnitude and components at the guessed location
                B = B_field(xs,ys,zs,trilinear)
                
                jac = jacobian(xs,ys,zs,trilinear)
                
                if np.linalg.det(jac)==0:
                    print('The matrix is singular')
                    break
                
                
                else:
                        
                        
                    jac_inv = inv(jacobian(xs,ys,zs,trilinear))
                        
                    xs_prev = xs
                    ys_prev = ys
                    zs_prev = zs
                    
                    xs = xs_prev-(jac_inv[0,0]*B[1]+jac_inv[0,1]*B[2]+jac_inv[0,2]*B[3])
                    ys = ys_prev-(jac_inv[1,0]*B[1]+jac_inv[1,1]*B[2]+jac_inv[1,2]*B[3])
                    zs = zs_prev-(jac_inv[2,0]*B[1]+jac_inv[2,1]*B[2]+jac_inv[2,2]*B[3])
            
                    new_B = B_field(xs,ys,zs,trilinear)
                    
                    step_x.append(xs-xs_prev)
                    step_y.append(ys-ys_prev)
                    step_z.append(zs-zs_prev)
                    
                    x.append(xs_prev+step_x[-1])
                    y.append(ys_prev+step_y[-1])
                    z.append(zs_prev+step_z[-1])
                    
                    err_rel_grid.append(math.sqrt((step_x[-1]/delta_x)**2+(step_y[-1]/delta_y)**2+(step_z[-1]/delta_z)**2))
                    err_rel_sol.append(math.sqrt((step_x[-1]/x[-1])**2+(step_y[-1]/y[-1])**2+(step_z[-1]/z[-1])**2))
                    
                    
                    if np.max([err_rel_grid[-1], err_rel_sol[-1]]) < tolerance:
                        converged = True
                    
                        B1 = math.sqrt(B_x1[i,j,k]**2 + B_y1[i,j,k]**2 + B_z1[i,j,k]**2)
                        B2 = math.sqrt(B_x1[i+1,j,k]**2 + B_y1[i+1,j,k]**2 + B_z1[i+1,j,k]**2)
                        B3 = math.sqrt(B_x1[i,j+1,k]**2 + B_y1[i,j+1,k]**2 + B_z1[i,j+1,k]**2)
                        B4 = math.sqrt(B_x1[i+1,j+1,k]**2 + B_y1[i+1,j+1,k]**2 + B_z1[i+1,j+1,k]**2)
                        B5 = math.sqrt(B_x1[i,j,k+1]**2 + B_y1[i,j,k+1]**2 + B_z1[i,j,k+1]**2)
                        B6 = math.sqrt(B_x1[i+1,j,k+1]**2 + B_y1[i+1,j,k+1]**2 + B_z1[i+1,j,k+1]**2)
                        B7 = math.sqrt(B_x1[i,j+1,k+1]**2 + B_y1[i,j+1,k+1]**2 + B_z1[i,j+1,k+1]**2)
                        B8 = math.sqrt(B_x1[i+1,j+1,k+1]**2 + B_y1[i+1,j+1,k+1]**2 + B_z1[i+1,j+1,k+1]**2)
                        
                        if n_steps>100:
                            print('Maximum number of steps exceeded -- exiting')
                            
                        if converged:
                            if ((xv[i] <= xs <= xv[i+1]) and (yv[j] <= ys <= yv[j+1]) and (zv[k] <= zs <= zv[k+1])):
                                if new_B[0] < tolerance*np.mean([B1,B2,B3,B4,B5,B6,B7,B8]):
                                    num_nulls+=1
                                    # here if we want, we can also get the eigenvectors/eigenvalues
                                    # use your previous function to get jacobian of magnetic field
                                    # use numpy.linalg.eig to find eigen-stuff of jacobian
                                    if zs <= zv[-2]: # this excludes the null points located on the null line that goes around the two outermost shells
                                        this_null = {'i':i, 'j':j, 'k':k, 'n': num_nulls, 'x': xs, 'y': ys, 'z': zs, 'B': new_B[0], 'Error' : np.array([err_rel_grid[-1], err_rel_sol[-1]]).max(), 'iter' : n_steps }
                                        null_list.append(this_null)
                                        
    return(null_list)


# function that checks if Bx/By/Bz changes sign:
# it compares the length of the list with the occurrence of each sign
# if '1' (positive) appears 8 times, then B has the same sign at all 8 corners
# similarly for -1 (negative) and 0 (field component = 0)
def check_sign(vertices):
   if len(vertices) < 1:
       return True
   return len(vertices) == vertices.count(vertices[0])

def field_sign_change (f):
 
    # returns a mask of dim (nx-1, ny-1, nz-1).
    # true implies that the component changes signs at one of the vertices of the rhs cell.
   
    p000 = (np.roll(f, (-0,-0,-0), axis=(0,1,2)) > 0)
    p100 = (np.roll(f, (-1,-0,-0), axis=(0,1,2)) > 0)
    p010 = (np.roll(f, (-0,-1,-0), axis=(0,1,2)) > 0)
    p110 = (np.roll(f, (-1,-1,-0), axis=(0,1,2)) > 0)
    p001 = (np.roll(f, (-0,-0,-1), axis=(0,1,2)) > 0)
    p101 = (np.roll(f, (-1,-0,-1), axis=(0,1,2)) > 0)
    p011 = (np.roll(f, (-0,-1,-1), axis=(0,1,2)) > 0)
    p111 = (np.roll(f, (-1,-1,-1), axis=(0,1,2)) > 0)
   
    all_pos = (  p000 &  p100 &  p010 &  p110 &  p001 &  p101 &  p011 &  p111 )[:-1,:-1,:-1]
    all_neg = ( ~p000 & ~p100 & ~p010 & ~p110 & ~p001 & ~p101 & ~p011 & ~p111 )[:-1,:-1,:-1]
   
    fsc = ( ~all_pos & ~all_neg )
   
    return(fsc)


# this function returns the trilinear coefficients for a particular field component 'B'
# u, v, w are the x, y, z coordinates with respective indices i,j,k
def trilinear_coeffs(u, v, w ,i, j, k, B):
    a = (1/((u[i]-u[i+1])*(v[j]-v[j+1])*(w[k]-w[k+1])))*(B[i,j,k+1]*u[i+1]*v[j+1]*w[k]+B[i,j+1,k]*u[i+1]*v[j]*w[k+1]+B[i+1,j,k]*u[i]*v[j+1]*w[k+1]+B[i+1,j+1,k+1]*u[i]*v[j]*w[k]-B[i,j,k]*u[i+1]*v[j+1]*w[k+1]-B[i,j+1,k+1]*u[i+1]*v[j]*w[k]-B[i+1,j,k+1]*u[i]*v[j+1]*w[k]-B[i+1,j+1,k]*u[i]*v[j]*w[k+1])
    b = (1/((u[i]-u[i+1])*(v[j]-v[j+1])*(w[k]-w[k+1])))*(B[i,j,k]*v[j+1]*w[k+1]-B[i,j,k+1]*v[j+1]*w[k]-B[i,j+1,k]*v[j]*w[k+1]+B[i,j+1,k+1]*v[j]*w[k]-B[i+1,j,k]*v[j+1]*w[k+1]+B[i+1,j,k+1]*v[j+1]*w[k]+B[i+1,j+1,k]*v[j]*w[k+1]-B[i+1,j+1,k+1]*v[j]*w[k])
    c = (1/((u[i]-u[i+1])*(v[j]-v[j+1])*(w[k]-w[k+1])))*(B[i,j,k]*u[i+1]*w[k+1]-B[i,j,k+1]*u[i+1]*w[k]-B[i,j+1,k]*u[i+1]*w[k+1]+B[i,j+1,k+1]*u[i+1]*w[k]-B[i+1,j,k]*u[i]*w[k+1]+B[i+1,j,k+1]*u[i]*w[k]+B[i+1,j+1,k]*u[i]*w[k+1]-B[i+1,j+1,k+1]*u[i]*w[k])
    d = (1/((u[i]-u[i+1])*(v[j]-v[j+1])*(w[k]-w[k+1])))*(-B[i,j,k]*w[k+1]+B[i,j,k+1]*w[k]+B[i,j+1,k]*w[k+1]-B[i,j+1,k+1]*w[k]+B[i+1,j,k]*w[k+1]-B[i+1,j,k+1]*w[k]-B[i+1,j+1,k]*w[k+1]+B[i+1,j+1,k+1]*w[k])
    e = (1/((u[i]-u[i+1])*(v[j]-v[j+1])*(w[k]-w[k+1])))*(B[i,j,k]*u[i+1]*v[j+1]-B[i,j,k+1]*u[i+1]*v[j+1]-B[i,j+1,k]*u[i+1]*v[j]+B[i,j+1,k+1]*u[i+1]*v[j]-B[i+1,j,k]*u[i]*v[j+1]+B[i+1,j,k+1]*u[i]*v[j+1]+B[i+1,j+1,k]*u[i]*v[j]-B[i+1,j+1,k+1]*u[i]*v[j])
    f = (1/((u[i]-u[i+1])*(v[j]-v[j+1])*(w[k]-w[k+1])))*(-B[i,j,k]*v[j+1]+B[i,j,k+1]*v[j+1]+B[i,j+1,k]*v[j]-B[i,j+1,k+1]*v[j]+B[i+1,j,k]*v[j+1]-B[i+1,j,k+1]*v[j+1]-B[i+1,j+1,k]*v[j]+B[i+1,j+1,k+1]*v[j])
    g = (1/((u[i]-u[i+1])*(v[j]-v[j+1])*(w[k]-w[k+1])))*(-B[i,j,k]*u[i+1]+B[i,j,k+1]*u[i+1]+B[i,j+1,k]*u[i+1]-B[i,j+1,k+1]*u[i+1]+B[i+1,j,k]*u[i]-B[i+1,j,k+1]*u[i]-B[i+1,j+1,k]*u[i]+B[i+1,j+1,k+1]*u[i])
    h = (1/((u[i]-u[i+1])*(v[j]-v[j+1])*(w[k]-w[k+1])))*(B[i,j,k]-B[i,j,k+1]-B[i,j+1,k]+B[i,j+1,k+1]-B[i+1,j,k]+B[i+1,j,k+1]+B[i+1,j+1,k]-B[i+1,j+1,k+1])
    tri_c = [a,b,c,d,e,f,g,h]
    return(tri_c)


# this function returns the magnetic field at a point location and its components
    
def B_field(x,y,z,k): # k is the array of trilinear coefficients
    #trilinear extrapolation
    bx = k[0,0] + k[0,1]*x +k[0,2]*y + k[0,3]*x*y + k[0,4]*z + k[0,5]*x*z + k[0,6]*y*z + k[0,7]*x*y*z
    by = k[1,0] + k[1,1]*x +k[1,2]*y + k[1,3]*x*y + k[1,4]*z + k[1,5]*x*z + k[1,6]*y*z + k[1,7]*x*y*z
    bz = k[2,0] + k[2,1]*x +k[2,2]*y + k[2,3]*x*y + k[2,4]*z + k[2,5]*x*z + k[2,6]*y*z + k[2,7]*x*y*z
                
    #magnitude of B field at the location
    magnitude = math.sqrt(bx*bx+by*by+bz*bz) 
    
    b = [magnitude,bx,by,bz]
    
    return(b)

    
    
# this function returns the jacobian matrix calculated at a point location    
def jacobian(x,y,z,k):
    
    dbxdx = k[0,1] + k[0,3]*y + k[0,5]*z + k[0,7]*y*z
    dbxdy = k[0,2] + k[0,3]*x + k[0,6]*z + k[0,7]*x*z
    dbxdz = k[0,4] + k[0,5]*x + k[0,6]*y + k[0,7]*x*y
                
    dbydx = k[1,1] + k[1,3]*y + k[1,5]*z + k[1,7]*y*z
    dbydy = k[1,2] + k[1,3]*x + k[1,6]*z + k[1,7]*x*z
    dbydz = k[1,4] + k[1,5]*x + k[1,6]*y + k[1,7]*x*y
                
    dbzdx = k[2,1] + k[2,3]*y + k[2,5]*z + k[2,7]*y*z
    dbzdy = k[2,2] + k[2,3]*x + k[2,6]*z + k[2,7]*x*z
    dbzdz = k[2,4] + k[2,5]*x + k[2,6]*y + k[2,7]*x*y
                
                
    jac = np.array([[dbxdx,dbxdy,dbxdz],[dbydx,dbydy,dbydz],[dbzdx,dbzdy,dbzdz]])

    return(jac)
    
    
    
# the following 3 functions determine the bilinear coefficients according to the face that is being analysed
# coord = face that is being analysed (i.e face with coordinate x = ...)
# j and k are the trilinear coefficients for the two field components used for the intersection
    
def x_face(coord,j,k):
    
    a1 = j[0] + j[1]*coord
    a2 = k[0] + k[1]*coord
    
    b1 = j[2] + j[3]*coord
    b2 = k[2] + k[3]*coord
    
    c1 = j[4] + j[5]*coord
    c2 = k[4] + k[5]*coord
    
    d1 = j[6] + j[7]*coord
    d2 = k[6] + k[7]*coord
    
    coeff = np.array([[a1,b1,c1,d1],[a2,b2,c2,d2]])
    
    return(coeff)

def y_face(coord,j,k):
    
    a1 = j[0] + j[2]*coord
    a2 = k[0] + k[2]*coord
    
    b1 = j[1] + j[3]*coord
    b2 = k[1] + k[3]*coord
    
    c1 = j[4] + j[6]*coord
    c2 = k[4] + k[6]*coord
    
    d1 = j[5] + j[7]*coord
    d2 = k[5] + k[7]*coord
    
    coeff = np.array([[a1,b1,c1,d1],[a2,b2,c2,d2]])
    
    return(coeff)
    
def z_face(coord,j,k):
    
    a1 = j[0] + j[4]*coord
    a2 = k[0] + k[4]*coord
    
    b1 = j[1] + j[5]*coord
    b2 = k[1] + k[5]*coord
    
    c1 = j[2] + j[6]*coord
    c2 = k[2] + k[6]*coord
    
    d1 = j[3] + j[7]*coord
    d2 = k[3] + k[7]*coord
    
    coeff = np.array([[a1,b1,c1,d1],[a2,b2,c2,d2]])
    
    return(coeff)


# this function returns the roots of a quadratic equation    
# k is the array of bilinear coefficients
def quad_roots (k):
    
    a = k[0,1]*k[1,3]-k[1,1]*k[0,3]
    b = k[0,0]*k[1,3]-k[1,0]*k[0,3]+k[0,1]*k[1,2]-k[1,1]*k[0,2]
    c = k[0,0]*k[1,2]-k[1,0]*k[0,2]
    
    if (b*b-4*a*c)>0 and a !=0:
        
        root1 = (-b+math.sqrt(b*b-4*a*c))/(2*a)
        root2 = (-b-math.sqrt(b*b-4*a*c))/(2*a)
    
        sol = np.array([root1,root2])
    
        return(sol)

# given the intersection of two field components, this function returns the sign of the thrid component at the two roots
# bi_coeff = bilinear coefficients for the intersecting field lines
# u and v are the directions along which we consider the intersection (i.e. bx = by = 0 --> u = x and v = y)
# u_i and v_i are the indeces associated with u and v (i,j,k)     
# face is the face of the cell that we consider for the analysis
# tri_ is an array of the trilinear coefficients 
# k is a parameter that tells on which face we are ( k=0 for face = x, k = 1 for face = y, k = 2 for face = z)
def bilinear(bi_coeff, u, v, u_i, v_i, face, tri_,k):
    
    b_sign = []
    
    a = bi_coeff[0,1]*bi_coeff[1,3]-bi_coeff[1,1]*bi_coeff[0,3]
    b = bi_coeff[0,0]*bi_coeff[1,3]-bi_coeff[1,0]*bi_coeff[0,3]+bi_coeff[0,1]*bi_coeff[1,2]-bi_coeff[1,1]*bi_coeff[0,2]
    c = bi_coeff[0,0]*bi_coeff[1,2]-bi_coeff[1,0]*bi_coeff[0,2] 
    #bilinear test applies only if determinant is greater than zero and a is non zero
    if (b*b - 4*a*c) > 0 and a != 0:
        u_roots = quad_roots(bi_coeff)
        #check that each root lies withing the range given by the the two corners of the cell
        if u[u_i] <= u_roots[0] <= u[u_i+1]:
            foo = bi_coeff[0,0] + bi_coeff[0,1]*u_roots[0]
            bar = bi_coeff[0,2] + bi_coeff[0,3]*u_roots[0]
            if bar != 0:
                v1 = -foo/bar
                if v[v_i] <= v1 <= v[v_i+1]:
                    if k == 0:
                        #calculate third components magnitude at the first root by using trilinear expression
                        b1 = tri_[0] + tri_[1]*face + tri_[2]*u_roots[0] + tri_[3]*face*u_roots[0] + tri_[4]*v1 + tri_[5]*v1*face + tri_[6]*u_roots[0]*v1 + tri_[7]*u_roots[0]*face*v1
                        b_sign.append(np.sign(b1))
                    if k == 1:
                        b1 = tri_[0] + tri_[1]*u_roots[0] + tri_[2]*face + tri_[3]*face*u_roots[0] + tri_[4]*v1 + tri_[5]*v1*u_roots[0] + tri_[6]*face*v1 + tri_[7]*u_roots[0]*face*v1
                        b_sign.append(np.sign(b1))
                    if k == 2:
                        b1 = tri_[0] + tri_[1]*u_roots[0] + tri_[2]*v1 + tri_[3]*u_roots[0]*v1 + tri_[4]*face + tri_[5]*u_roots[0]*face + tri_[6]*v1*face + tri_[7]*u_roots[0]*v1*face
                        b_sign.append(np.sign(b1))
        elif u[u_i] <= u_roots[1] <= u[u_i+1]:
            foo = bi_coeff[1,0] + bi_coeff[1,1]*u_roots[1]
            bar = bi_coeff[1,2] + bi_coeff[1,3]*u_roots[1]
            if bar != 0:
                v2 = -foo/bar
                if v[v_i] <= v2 <= v[v_i+1]:
                    if k == 0:
                        b2 = tri_[0] + tri_[1]*face + tri_[2]*u_roots[1] + tri_[3]*face*u_roots[1] + tri_[4]*v2 + tri_[5]*v2*face + tri_[6]*u_roots[1]*v2 + tri_[7]*u_roots[1]*face*v2
                        b_sign.append(np.sign(b2))
                    if k == 1:
                        b2 = tri_[0] + tri_[1]*u_roots[1] + tri_[2]*face + tri_[3]*face*u_roots[1] + tri_[4]*v2 + tri_[5]*v2*u_roots[1] + tri_[6]*face*v2 + tri_[7]*u_roots[1]*face*v2
                        b_sign.append(np.sign(b2))
                    if k == 2:
                        b2 = tri_[0] + tri_[1]*u_roots[1] + tri_[2]*v2 + tri_[3]*u_roots[1]*v2 + tri_[4]*face + tri_[5]*u_roots[1]*face + tri_[6]*v2*face + tri_[7]*u_roots[1]*v2*face
                        b_sign.append(np.sign(b2))
                        
    return(b_sign)
    



