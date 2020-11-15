# Null_Finder

Magnetic reconnection is considered to play a key role in the energy release processes
of astrophysical plasmas, therefore in order to expand our understanding of the behaviour
of plasmas such as the slow solar wind (SSW) we need to identify potential sites for
reconnection to occur. For this purpose, solar physicists have developed several models
to approximate the weak coronal field of the Sun among which there are potential fields (PFSS),
that correspond to the simplest equilibria, and non-linear force-free fields (NLFF), which
account for heliospheric current sheets and hence have a higher degree of complexity. 
One fundamental question to address is how the topology of the structures of the former
differs from the one found in the latter. 

In this regard, a null finder has been built: this Python code can be run for any
magnetic field model and employs a trilinear interpolation and the Newton-Raphson
root-finding algorithm to return a list of magnetic null points. 


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
