# -*- coding: utf-8 -*-
"""
Created on Mon Apr  27 12:00:00 2020

@author: Ivan Kuznetsov aka kuivi
"""
#
# This file is part of pyfesom2: https://github.com/FESOM/pyfesom2.git
# Original code by Ivan Kuznetsov, 2020, folowing FESOM2 code structure from Dmitry Sidorenko, 2013
#

import pandas as pd
import numpy as np
from netCDF4 import Dataset
import os
import logging
import time
from netCDF4 import num2date
import matplotlib.pyplot as plt
import joblib
import pickle

import importlib
# construct dictionary with colormaps for variables
cmocean_spec = importlib.util.find_spec("cmocean")
if (cmocean_spec is not None):
    import cmocean
    cmap_dict = {
     'temperature':cmocean.cm.thermal,
     'temp':cmocean.cm.thermal,
     'salinity':cmocean.cm.haline,
     'salt':cmocean.cm.haline,
     'ice':cmocean.cm.ice,
     'dense':cmocean.cm.dense,
     'u':cmocean.cm.speed,
     'v':cmocean.cm.speed,
     'w':cmocean.cm.speed,
     'U':cmocean.cm.speed,
     'V':cmocean.cm.speed,
     'W':cmocean.cm.speed,
     'z':cmocean.cm.topo,
     'zbar':cmocean.cm.topo
     }
else:
    cmap = plt.get_cmap('viridis')  
    cmap_dict = {
     'temperature':cmap,
     'temp':cmap,
     'salinity':cmap,
     'salt':cmap,
     'ice':cmap,
     'dense':cmap,
     'u':cmap,
     'v':cmap,
     'w':cmap,
     'U':cmap,
     'V':cmap,
     'W':cmap,
     'z':cmap,
     'zbar':cmap
     }    
        

def load_c_mesh(path, exp = "", usepickle=False, usejoblib=False, protocol=4):
    """ Loads FESOM-C mesh

    Parameters
    ----------
    path : str
        Path to the directory with mesh files
    usepickle (optional): bool
        use pickle file to store or load mesh data
    usejoblib (optional): bool
        use joblib file to store or load mesh data
    protocol (optional): int
        used for pickle, only way to save data more than 4 Gb
    Returns
    -------
    mesh : object
        fesom_c_mesh object
    """
    path = os.path.abspath(path)
    if (usepickle == True) and (usejoblib == True):
        raise ValueError(
            "Both `usepickle` and `usejoblib` set to True, select only one"
        )

    if usepickle:
        pickle_file = os.path.join(path, "pickle_mesh_py3_fesom_c")
        print(pickle_file)

    if usejoblib:
        joblib_file = os.path.join(path, "joblib_mesh_fesom_c")

    if usepickle and (os.path.isfile(pickle_file)):
        print("The usepickle == True)")
        print("The pickle file for FESOM_C exists.")
        print("The mesh will be loaded from {}".format(pickle_file))

        ifile = open(pickle_file, "rb")
        mesh = pickle.load(ifile)
        ifile.close()
        return mesh

    elif (usepickle == True) and (os.path.isfile(pickle_file) == False):
        print("The usepickle == True")
        print("The pickle file for FESOM_C DO NOT exists")
        print("The mesh will be saved to {}".format(pickle_file))

        mesh = fesom_c_mesh(path=path, exp=exp)
        logging.info("Use pickle to save the mesh information")
        print("Save mesh to binary format")
        outfile = open(pickle_file, "wb")
        pickle.dump(mesh, outfile, protocol=protocol)
        outfile.close()
        return mesh

    elif (usepickle == False) and (usejoblib == False):
        mesh = fesom_c_mesh(path=path, exp=exp)
        return mesh

    if (usejoblib == True) and (os.path.isfile(joblib_file)):
        print("The usejoblib == True)")
        print("The joblib file for FESOM_C exists.")
        print("The mesh will be loaded from {}".format(joblib_file))

        mesh = joblib.load(joblib_file)
        return mesh

    elif (usejoblib == True) and (os.path.isfile(joblib_file) == False):
        print("The usejoblib == True")
        print("The joblib file for FESOM_C DO NOT exists")
        print("The mesh will be saved to {}".format(joblib_file))

        mesh = fesom_c_mesh(path=path, exp=exp)
        logging.info("Use joblib to save the mesh information")
        print("Save mesh to binary format")
        joblib.dump(mesh, joblib_file)

        return mesh

def load_c_station(fname):
    """ Loads FESOM-C station

    Parameters
    ----------
    fname : str
        Path to the directory with mesh files

    Returns
    -------
    mesh : object
        fesom_c_station object
    """
    fname = os.path.abspath(fname)
    station = fesom_c_station(fname)
    return station



    
    
class fesom_c_mesh(object):
    """ Creates instance of the FESOM-C mesh.
    This class creates instance that contain information
    about FESOM-C mesh. At present the class works with
    ASCII representation of the FESOM-C grid, 
    it read also netCDF version of FESOM-C.
    while reading ASCII version no information about sigma is loaded.
    NetCDF is preferably. 

    Minimum requirement is to provide the path to the directory and 
    <expname> experiment name,
    where following files should be located :

    - <expname>_nod2d.out
    - <expname>_elem2d.out
    - <expname>_depth.out

    Parameters
    ----------
    path : str
        Path to the directory with mesh files 
           OR
        if path ends on ".nc", netcdf file will be used to read mesh

    exp : str (optional for netcdf file)
        name of the experiment (<exp>_nod2d.out)
        if no exp is provided for ascii case or exp = "" than files:
            nod2d.out, ... will be used
        

    Attributes
    ----------
    path : str
        Path to the directory with mesh files
    x2 : array
        x position (lon) of the surface node
    y2 : array
        y position (lat) of the surface node
    n2d : int
        number of 2d nodes
    e2d : int
        number of 2d elements (triangles)
    type : str
        type of mesh (fesom-c for FESOM-C)
        
    Returns
    -------
    mesh : object
        fesom_mesh object
    """

    def __init__(self, path, exp=""):
        #add type of mesh
        self.type = 'fesom_c'

        #find if nectdf file is provided 
        if (path[-3:] != '.nc'):
            useascii = True
            usenetcdf = False
            self.path = os.path.abspath(path)
        else:
            useascii = False
            usenetcdf = True
            s = os.path.abspath(path)
            self.path = os.path.dirname(s)
        
        if not os.path.exists(path):
            raise IOError('The path/file "{}" does not exists'.format(path))
        #predifinition. (why?)
        self.e2d = 0
        self.nlev = 0
        self.zlevs = []
        self.topo = []

        if useascii:    
            s = ""
            if (exp != ""):
                s=exp+"_"
            self.nod2dfile = os.path.join(self.path, s+"nod2d.out")
            self.elm2dfile = os.path.join(self.path, s+"elem2d.out")
            self.depth2dfile = os.path.join(self.path, s+"depth.out")
                
            if not os.path.exists(self.nod2dfile):
                raise IOError('The file "{}" does not exists'.format(self.nod2dfile))
            if not os.path.exists(self.elm2dfile):
                raise IOError('The file "{}" does not exists'.format(self.elm2dfile))
            if not os.path.exists(self.depth2dfile):
                raise IOError('The file "{}" does not exists'.format(self.depth2dfile))
        else:
            self.ncfile = os.path.join(path)            

        logging.info("load 2d part of the mesh")
        start = time.clock()
        if useascii:
            self.read2d()
        else:
            self.read2d_nc()
        end = time.clock()
        print("Load 2d part of the mesh in {} second(s)".format(str(int(end - start))))

    def read2d(self):
        # funcion to read mesh files for FESOM-C branch, 
        # * it has 4 nodes elements in each element
        # * it uses sigma vertical discretization
        # * no aux file, but depth
        file_content = pd.read_csv(
            self.nod2dfile,
            delim_whitespace=True,
            skiprows=1,
            names=["node_number", "x", "y", "flag"],
        )
        self.x2 = file_content.x.values
        self.y2 = file_content.y.values
        self.ind2d = file_content.flag.values
        self.n2d = len(self.x2)

        file_content = pd.read_csv(
            self.elm2dfile,
            delim_whitespace=True,
            skiprows=1,
            names=["first_elem", "second_elem", "third_elem", "fourth_elem"],
        )
        self.elem = file_content.values - 1
        self.e2d = np.shape(self.elem)[0]
        
        #read depths
        file_content = pd.read_csv(
            self.depth2dfile,
            delim_whitespace=True,
            skiprows=0,
            names=["depth"],
        )
        self.topo = file_content.depth.values
               

        ###########################################
        # computation of volumes skiped here (it is done in FESOM-C output nc files)
        # compute the 2D lump operator skiped for fesom_c for now

        return self

    def read2d_nc(self):
        # funcion to read mesh files for FESOM-C branch. NetCDF version 
        # * it has 4 nodes elements in each element
        # * it uses sigma vertical discretization
        # * no aux file, but depth
        # variables will be added to object (mesh) if availeble in nc file,
        # personaly I do not like defind so many variable like self.<name>
        # i would prefer self.aux = {}
        #    self.aux.update({"varname":value})
        #    so to call it: mesh.aux['area']
        #    how does it works with paralel libs ?
        def loadvar(var):
            if (var in ncf.variables):
                data = ncf.variables[var][:].data
            return data
            
        ncf = Dataset(self.ncfile)
        
        self.x2 = loadvar('lon')
        self.y2 = loadvar('lat')
        self.elem = loadvar('nv')
        self.topo =  loadvar('depth')
        self.x2_e = loadvar('lon_elem')
        self.y2_e = loadvar('lat_elem')
        self.sigma_lev = loadvar('sigma_lev')
        self.topo_e = loadvar('depth_elem')
        self.area = loadvar('area')
        self.elem_area = loadvar('elem_area')
        self.w_cv = loadvar('w_cv')
        self.nod_in_elem2d_num = loadvar('nod_in_elem2d_num')
        self.nod_in_elem2d = loadvar('nod_in_elem2d')
        ncf.close()

        return self

class fesom_c_station(object):
    """ Creates instance of the FESOM-C station.
    This class creates instance that contain information
    about station, functions for read data from station and plot data.
    It works only with NetCDF files type 1d FESOM-C.

    Minimum requirement is to provide the path to the file tieh station.

    Parameters
    ----------
    fname : str
        Path to the file with station.

    Attributes
    ----------
    fname : str
        Path to the directory with mesh files
    x2 : float
        x position (lon) of the surface node
    y2 : float
        y position (lat) of the surface node
    ....    
        
    Returns
    -------
    station : object
        fesom_c station object
    """    
    def __init__(self, fname):
        self.fname = os.path.abspath(fname)
        self.path  = os.path.dirname(fname)
        
        if not os.path.exists(fname):
            raise IOError('The file "{}" does not exists'.format(fname))

        self.readstat()
        
    def readstat(self):
        # read basic stats from station
        def loadvar(var):
            if (var in ncf.variables):
                data = ncf.variables[var][:].data
            return data
        aux = ['time','sigma_lev','depth','lon','lat','nv','depth','lon_elem','lat_elem',
               'elem_area','depth_elem','area','w_cv','nod_in_elem2d_num','nod_in_elem2d']
        #aux_add = ['fvcoastal_mesh']
        fname = self.fname
        ncf = Dataset(fname,'r')
        mtime_raw = ncf.variables['time'][:]
        a = ncf.variables['time'].getncattr('units')
        self.mtime = num2date(mtime_raw,a)   
        self.mtimec = np.array([t.timestamp() for t in self.mtime])
        self.sigma_lev = loadvar('sigma_lev') # sigma levels
        self.nlev = len(self.sigma_lev)
        self.depth = loadvar('depth') #depth on node
        self.z0 = self.depth*(1-self.sigma_lev) #depth of sigma level on node
        self.z0m1 = (self.z0[0:-1]+self.z0[1:])/2.0 #depth of sigma-m1 level on node
        self.depth_elem = loadvar('depth_elem')  #depth on element
        self.z0e = self.depth_elem*(1-self.sigma_lev)  #depth of sigma level on element      
        self.z0em1 = (self.z0e[0:-1]+self.z0e[1:])/2.0 #depth of sigma-1 level on element
        self.x2 = loadvar('lon')
        self.y2 = loadvar('lat')
        self.elem = loadvar('nv')
        self.x2_e = loadvar('lon_elem')
        self.y2_e = loadvar('lat_elem')
        self.area = loadvar('area')
        self.elem_area = loadvar('elem_area')
        self.w_cv = loadvar('w_cv')
        self.nod_in_elem2d_num = loadvar('nod_in_elem2d_num')
        self.nod_in_elem2d = loadvar('nod_in_elem2d')        
        varlist = ncf.variables.keys()
        self.varlist = varlist - aux
        ncf.close()    

        return self        
        
    def get_datat2d(self,var):
        # mesh is a station, will read "var" variable
        # no interpolation is done here
        ncf = Dataset(self.fname,'r')                   #-- add data file
        if (ncf.variables[var].ndim != 3):
            raise IOError('The variable "{}" is not 2d variable.'.format(var)) 
        data = ncf.variables[var][:,0,:].data
        data = data.transpose()
        ncf.close()    
        return data

    def get_varparam(self,var):
        # mesh is a station, will read "var" standard_name and units
        ncf = Dataset(self.fname,'r')
        attrs = ncf.variables[var].ncattrs()
        if ('standard_name' in attrs):
            standard_name = ncf.variables[var].getncattr('standard_name')
        else:
            standard_name = var
        if ('units' in attrs):            
            units = '['+ncf.variables[var].getncattr('units')+']'
        else:
            units = '[]'
        ncf.close()    
        return {'standard_name':standard_name, 'units':units}
    
    def contourfz0(self,var):
        # read "var" variable
        # contourf (Hovmöller) var
        # no interpolation is done here
        if not (var in self.varlist):
            raise IOError('The variable "{}" is not in station nc file.'.format(var)) 
            
        data = self.get_datat2d(var)
        dparam = self.get_varparam(var)
        if (data.shape[0] == self.nlev):
            z = self.z0
        elif (data.shape[0] == self.nlev -1):
            z = self.z0m1
        else:
            raise IOError('The data shape is not Nsigma or Nsigma-1 shape="{}"'.format(data.shape)) 
        cmap = select_cmap(var)    
        fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 10), dpi=90)
        od = axes.contourf(self.mtime, z, data,
                  cmap=cmap)
#                  levels=np.linspace(cmin, cmax, 100),
#                 vmin=cmin)   
#   od.set_clim(cmin,cmax)
#        axes.set_xlim(t.min(),t.max())
#        axes.set_ylim(mz.max().round(),0)
        axes.grid(color='k', alpha=0.5, linestyle='--')
        axes.set_xlabel('Time',fontsize=18)    
        axes.tick_params(labelsize=18)
        axes.set_ylabel('Depth, [m]',fontsize=18)    
        #axes[0].xaxis.set_tick_params(labelsize=fs)
        #axes[0].yaxis.set_tick_params(labelsize=fs)
        cbar = fig.colorbar(od, aspect=40)
#                        ticks=np.linspace(np.round(cmin,1),np.round(cmax,1),7))  
        cbar.ax.tick_params(labelsize=18)
        axes.set_title(dparam['standard_name']+', '+dparam['units'],fontsize=18)
        axes.invert_yaxis()
        fig.autofmt_xdate()
        axes.autoscale_view()                
        return {'fig': fig, 'axes':axes, 'contourf':od}
    
    
def select_cmap(var):
    if (var in cmap_dict):
        return cmap_dict[var]
    else:
        return plt.get_cmap('viridis')  
    
    
    