import os as os
from math import pi
import subprocess as subp
import timeit 
import h5py

import numpy as np
from numpy.linalg import eig, inv

import scipy.ndimage.interpolation as spint
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter as gfilt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq 
from scipy.signal import argrelextrema
from scipy.stats import binned_statistic_2d as bstat2d
from scipy.signal import convolve2d as conv2d
import astropy.io.fits as pf
import astropy.convolution as convolve

# Illustris Data Reader
def illload(f):
    fdat = h5py.File(f,'r')
    ou   = np.zeros((len(fdat['PartType4']['Masses']),10))
    for i in range(3):
        ou[:,i]   = fdat['PartType4']['Coordinates'][:,i]/0.73
        ou[:,i+3] = fdat['PartType4']['Velocities'][:,i]
    ou[:,8] = fdat['PartType4']['Masses']
    ou[:,9] = 10.**((-0.4)*(fdat['PartType4']['GFM_StellarPhotometrics'][:,5]))

    return ou

# Illustris Data Reader (incl gas)
def illload_gas(f):
    fdat = h5py.File(f,'r')

    # Check if subhalo has gas
    if len(fdat['PartType0']) > 0:
        gflag = 1
    else:
        gflag = 0

    # Create output arrays to contain star and gas (if present)
    # particle data:
    # column  |  dtype  |  what is it? |  units
    # ------- | ------- | ------------ | -------
    # 0       | float   |  x-position  |  ?????
    # 1       | float   |  y-position  |  ?????
    # 2       | float   |  z-position  |  ?????
    # 3       | float   |  x-velocity  |  km/s
    # 4       | float   |  y-velocity  |  km/s
    # 5       | float   |  z-velocity  |  km/s
    # 6       | int     |  data flag   | 1 = stars, 2 = gas
    # 7       | n/a     |  empty       | (historical issue, placeholder)
    # 8       | float   |  mass        |  ?????
    # 9       | float   |  sf-props    |  birth time (star), SFR (gas, Modot/yr)
    ou_star   = np.zeros((len(fdat['PartType4']['Masses']),10))
    if gflag == 1: ou_gas    = np.zeros((len(fdat['PartType0']['Masses']),10))

    # Loop through xyz, add position and velocity info
    for i in range(3):
        ou_star[:,i]   = fdat['PartType4']['Coordinates'][:,i]
        ou_star[:,i+3] = fdat['PartType4']['Velocities'][:,i]
        if gflag == 1:
            ou_gas[:,i]   = fdat['PartType0']['Coordinates'][:,i]
            ou_gas[:,i+3] = fdat['PartType0']['Velocities'][:,i]

    # add mass and SF info (stars)
    ou_star[:,8] = fdat['PartType4']['Masses']
    ou_star[:,6] = 1
    ou_star[:,9] = fdat['PartType4']['GFM_StellarFormationTime']

    # add mass and SF info (gas, mass for HI ONLY!!)
    if gflag == 1:
        gmt = fdat['PartType0']['Masses']
        nhf = fdat['PartType0']['NeutralHydrogenAbundance']
        ou_gas[:,8] = np.array(gmt)*np.array(nhf)
        ou_gas[:,6] = 2
        ou_gas[:,9] = fdat['PartType0']['StarFormationRate']

        # concatenate star and gas outputs if gas present
        ou = np.concatenate((ou_star,ou_gas),axis=0)
    else:
        ou = ou_star
    return ou




# Vector functions -------------------------------
#   -inputs are numpy arrays
#   -most valid for 2D or 3D vectors

# get vector magnitude
def vector_mag(v):
    return np.sqrt(np.dot(v,v))

# get angle between 2 vectors
def vector_angle(v1,v2):
    return np.arccos(np.dot(v1,v2)/(vector_mag(v1)*vector_mag(v2)))

# create unit vector in direction of vector A 
def uvec(A):
    mag = np.sqrt(np.dot(A,A))
    return A/mag




    
# Rotation stuff  --------------------------------------------------
#  -used for rotating positions and velocities of particles in 3D

# Generate rotation matrix rotating by angle "ang" about unit vector "e"
def Rmat(ang,e):
    x,y,z = e[0],e[1],e[2]
    c,s   = np.cos(ang),np.sin(ang)

    R = np.array([[c+(x*x*(1.-c)), (x*y*(1.-c))-(z*s), (x*z*(1.-c))+(y*s)],
                  [(x*y*(1.-c))+(z*s), c+(y*y*(1.-c)), (y*z*(1.-c))-(x*s)],
                  [(z*x*(1.-c))-(y*s), (z*y*(1.-c))+(x*s), c+(z*z*(1.-c))]])

    return R

# Generate rotation matrix which rotates vector "A" to match vector "B"
def Rm_vecs(A,B):
    u   = uvec(A)
    v   = uvec(B)
    ang = (-1.)*vector_angle(u,v)
    F2 = np.cross(v,u)/vector_mag(np.cross(v,u))
    return Rmat(ang,F2)

# Single rotation using phi,theta,psi (use rand_vp iterations for uniform
#   viewpoints on the surface of a sphere)
def single_rot(x,y,z,xv,yv,zv,phi,theta,psi):
    
    x1   = np.array([x,y,z])
    v1     = np.array([x+xv,y+yv,z+zv])
    rmat1   = np.array([[np.cos(phi),-np.sin(phi),0.],
                        [np.sin(phi),np.cos(phi),0.],
                        [0.,0.,1.]])
    x2   = np.dot(rmat1,x1)
    v2 = np.dot(rmat1,v1)
    
    rmat2  = np.array([[1.,0.,0.],
                       [0.,np.cos(theta),np.sin(theta)],
                       [0.,-np.sin(theta),np.cos(theta)]])
    x3   = np.dot(rmat2,x2)
    v3 = np.dot(rmat2,v2)

    rmat3  = np.array([[np.cos(psi),-np.sin(psi),0.],
                        [np.sin(psi),np.cos(psi),0.],
                        [0.,0.,1.]])
    xf,v4 = np.dot(rmat3,x3),np.dot(rmat3,v3)
    vf = v4-xf
    return xf,vf




# Generate phi,theta,psi for uniform sampling on surface of a sphere,
#  for use with single_rot (above)
def rand_vp():
    u1 = np.random.uniform()
    u2 = np.random.uniform()

    phi = (2.*np.pi)*(np.random.uniform())
    theta = np.arccos((2.*u1)-1.)
    psi   = (2.*np.pi)*(u2)

    return phi,theta,psi



# Functions for measuring 3D shape  ---------------------------------
# get the ellipsoidal half mass radius assuming
#      axis ratios b* (b/a) and c* (c/a)
def new_hmr(x,y,z,b,c,m):
    hmass = np.sum(m)*.5
    
    tmr = np.sqrt((x*x)+((y/b)*(y/b))+((z/c)*(z/c)))
    sort_indices = np.argsort(tmr)
    cum_mass = np.cumsum(m[sort_indices])
    tt = np.where(abs(cum_mass - hmass) == np.min(abs(cum_mass - hmass)))
    
    return sort_indices[0:tt[0][0]]

# get the ellipsoidal "fractional mass radius"
#    e.g. 1/3rd mass radius set frac = 0.333
def frac_mr(x,y,z,b,c,m,frac):
    hmass = np.sum(m)*frac
    
    tmr = np.sqrt((x*x)+((y/b)*(y/b))+((z/c)*(z/c)))
    sort_indices = np.argsort(tmr)
    cum_mass = np.cumsum(m[sort_indices])
    tt = np.where(abs(cum_mass - hmass) == np.min(abs(cum_mass - hmass)))
    
    return sort_indices[0:tt[0][0]]

# Produce the weighted ellipsoid tensor, use mass or luminosity weighting.
# Only measures for selected particles where "sel" is an integer array of
# selected particle indices.
# bc = axis ratios --> [b/a,c/a]
def ellipsoid_tensor(m,xyz,bc,sel):
    n_particles = len(m)
    M = np.zeros((3,3))
    rad = np.sqrt((xyz[0]**2.)+((xyz[1]/bc[0])**2.)+((xyz[2]/bc[1])**2.))
    for alpha in range(3):
        for beta in range(3):
            M[alpha,beta] = np.sum(m[sel]*xyz[alpha,sel]*xyz[beta,sel]/rad[sel])

    return M

# Get p and q from eigenvalues of the ellipsoid tensor (above)
def ell_axes(m,xyz,bc,sel):
    M = ellipsoid_tensor(m,xyz,bc,sel)
    eigval,eigvec = np.linalg.eig(M)
    ti=np.where(eigval == np.max(eigval))
    tf=np.where(eigval == np.min(eigval))
    eigsort = sorted(eigval)
    tm=np.where(eigval == eigsort[1])
    eigsort = eigsort[::-1]
    p =  np.sqrt(eigsort[1]/eigsort[0])
    q =  np.sqrt(eigsort[2]/eigsort[0])

    zax = eigvec[:,tf[0][0]]
    yax = eigvec[:,tm[0][0]]

    return yax,zax,eigsort

# Get angular momentum vector
def omega(m,x,y,z,xv,yv,zv):
    r = np.array([x,y,z])
    v = np.array([xv,yv,zv])
    r = r.transpose()
    v = v.transpose()
    
    LL = np.cross(r,v)
    L_all = np.array([np.sum(m*LL[:,0]),np.sum(m*LL[:,1]),np.sum(m*LL[:,2])])
    return L_all

# measure p, q, and L for an individual galaxy returns:
#   1 average of the last 6 p and q measurements
#   2 the indices of the selected points used for the measurement
#   3 the xyz positions,
#   4 the number of iterations
#   5 the xyz velocities
#   6 the 3D angular momentum vector
def measure_pql(m,xyz,vxyz):
    # Setting up
    a,b,c = 1.,1.,1.
    vxyz = np.array([xyz[0]+vxyz[0],xyz[1]+vxyz[1],xyz[2]+vxyz[2]])
    e1 = np.array([1.,0.,0.])
    e2 = np.array([0.,1.,0.])
    e3 = np.array([0.,0.,1.])
    

    # Select half mass radius with initial guess of sphere (b=c=1)
    tmsel2  = new_hmr(xyz[0],xyz[1],xyz[2],b,c,m)
    qs2 = []
    ps2 = []
    fl = 0
    cnt= 0
    ind=[]
    tm_p,tm_q = 1.,1.
    while fl == 0:
        # count how many iterations have been run
        ind.append(cnt)
        cnt+=1

        # measure the ellipsoidal axes with current selection
        # and get p and q
        tmy,tmz,tme = ell_axes(m,xyz,[tm_p,tm_q],tmsel2)
        tm_p,tm_q = np.sqrt(tme[1]/tme[0]),np.sqrt(tme[2]/tme[0])
        qs2.append(tm_q)
        ps2.append(tm_p)

        # Check if current value is close to (or equal to) the last 10
        # iterations. If so return current p and q.
        if cnt > 10:
            lst_ps = np.abs(np.array(ps2[-9:])-tm_p)
            lst_qs = np.abs(np.array(qs2[-9:])-tm_q)
            if all(i < .01 for i in lst_ps) and all(j < .01 for j in lst_qs):
                fl+=1

        # Aport if iteration limit is reached, output current p and q.
        if cnt > 50:
            fl+=1

        # Check if current z-axis is in the same direction as
        # the unit vector (0,0,1). If not, rotate such that it is
        if np.dot(tmz,e3) != np.sqrt(np.dot(tmz,tmz)):
            Rm1 = Rm_vecs(tmz,e3)
            v21 = np.dot(Rm1,tmy)
            Rm2 = Rm_vecs(v21,e2)
            xyz = np.dot(Rm1,xyz)
            xyz = np.dot(Rm2,xyz)
            vxyz= np.dot(Rm1,vxyz)
            vxyz= np.dot(Rm2,vxyz)

        # Redo half mass radius selection with updated p and q
        tmsel2 = new_hmr(xyz[0],xyz[1],xyz[2],tm_p,tm_q,m)

    # Move velocity vector back to actual values
    vxyz = np.array([vxyz[0]-xyz[0],vxyz[1]-xyz[1],vxyz[2]-xyz[2]])

    # Measure angular momentum about fitted z-axis
    omega_vector = omega(m[tmsel2],xyz[0][tmsel2],xyz[1][tmsel2],xyz[2][tmsel2],vxyz[0][tmsel2],vxyz[1][tmsel2],vxyz[2][tmsel2])

    # Return the average of the last 6 p and q measurements, the indices of
    # the selected points used for the measurements, the xyz positions, the
    # number of iterations, the xyz velocities, and the angular momentum vector
    return np.mean(ps2[-6:-1]),np.mean(qs2[-6:-1]),tmsel2,xyz,cnt,vxyz,omega_vector




# Stuff for 2D projections   ---------------------------------
# Make the mass map, can output fits if file_out set to output file name
# pos2 = xyz positions
# mass = mass array
def particle_2_image(pos2,mass,dist_fact,mass_fact,pixelscale,image_size,fwhm,file_out=False):
    fwhm2 = fwhm/pixelscale
    cc = fwhm2/2.35482
    sax = int(round(image_size/pixelscale))
    bex = np.arange((-1.)*image_size,image_size,pixelscale)
    bey = np.arange((-1.)*image_size,image_size,pixelscale)

    mass_hist,xe1,ye1,be1 = bstat2d(pos2[2],pos2[1],mass*mass_fact,statistic='sum',bins=(bex,bey))
    kern = convolve.Gaussian2DKernel(cc)

    if not file_out: pass

    else:
        hdu = pf.PrimaryHDU(convolve.convolve(mass_hist,kern))
        hdu.writeto(file_out,overwrite=True)
                
    return convolve.convolve(mass_hist,kern)

def single_gauss(x,a1,xo,c):
    return a1*np.exp((-1.*((x-xo)**2.))/(2.*c*c))

def fit_gauss(bins,hist,p0):
    popt, pcov = curve_fit(single_gauss,bins,hist,p0=p0)
    return single_gauss(bins,*popt),popt

# Create the velocity and velocity dispersion maps
#@profile
def particle_2_kmaps(pos2,vel2,mass,good_pix_map,mass_map,dist_fact,vel_fact,pixelscale,image_size,fwhm,vel_res,file_out,cutval):
    fwhm2 = fwhm/pixelscale
    cc = np.sqrt(fwhm2)/2.35482
    sax = int(round(image_size/pixelscale))-1
    bex = np.arange((-1.)*image_size,image_size,pixelscale)
    bey = np.arange((-1.)*image_size,image_size,pixelscale)

    vel_hist = good_pix_map*0.-1000.
    dis_hist = good_pix_map*0.-1000.
    fit_massm = good_pix_map*0.-1000.
    unc_massm = good_pix_map*0.-1000.
    chi2      = good_pix_map*0.-1000.
    conv_b   = np.arange((-4.)*vel_res,4.*vel_res,24.)
    conv_sp  = single_gauss(conv_b,1.,0.,vel_res)

    for i in range(-sax,sax+1):
        for j in range(-sax,sax+1):
            tm_loc = [i+int(sax),j+int(sax)]

            if good_pix_map[tm_loc[0],tm_loc[1]] != 1000 and mass_map[tm_loc[0],tm_loc[1]] > cutval:
                pixsel = np.where((abs(pos2[2]-(float(i)*pixelscale)) <= pixelscale/2.)&(abs(pos2[1]-(float(j)*pixelscale)) <= pixelscale/2.))[0]
                
                tself = np.where((abs(pos2[2]-(float(i)*pixelscale)) < 5.*cc)&(abs(pos2[1]-(float(j)*pixelscale)) < 5.*cc))[0]
                xdt = (pos2[2][tself]-(float(i)*pixelscale))**2.
                ydt = (pos2[1][tself]-(float(j)*pixelscale))**2.
                ddt = np.sqrt(xdt+ydt)
                veloci_tmp = vel2[0][tself]
                weight_tmp = (mass[tself])*np.exp(-1.*(ddt*ddt)/(2.*(cc**2.)))

                if len(tself > 0):
                    zdt = pos2[0][tself]
                    zme = np.average(zdt,weights=weight_tmp)
                    zspr= (np.max(zdt)-np.min(zdt))/100.
                    zsl = np.where(abs(zdt-zme) <= zspr)
                    tself2 = (abs(pos2[2]-(float(i)*pixelscale)) < 3.*cc)&(abs(pos2[1]-(float(j)*pixelscale)) < 3.*cc)&(abs(pos2[0]-zme) <= zspr)
                    tself2 = np.where(tself2 == True)
                    tself2 = tself2[0]
                
                    xdt2 = (pos2[2][tself2]-(float(i)*pixelscale))**2.
                    ydt2 = (pos2[1][tself2]-(float(j)*pixelscale))**2.
                    ddt2 = np.sqrt(xdt2+ydt2)
                
                    wez    = (mass[tself2])*np.exp(-1.*(ddt2*ddt2)/(2.*(cc**2.)))

                
                    bs = np.arange(-1000,1000,24.)
                    htmp,be = np.histogram(vel2[0][tself],bins=bs,weights=weight_tmp)
                    bw = bs[1]-bs[0]
                    bt = be[:-1]+bw/2.

                    tt = np.where(htmp == np.max(htmp))
                    p0 = [htmp[tt[0][0]],bt[tt[0][0]],50.]

                    cf=0.1
                    cut1 = cutval
                    #f1t = np.where(htmp < cut1)
                    #htmp[f1t[0]] = cut1
                    #htmp-=cut1
                    noi = np.random.normal(0,scale=cut1,size=len(bt))

                    try:
                        f1,p1 = fit_gauss(bt,htmp+noi,p0)
                    except:
                        p1 = [-1000.,-1000.,-1000.]
                        f1 = bt*0.
                    
                    vel_hist[tm_loc[0],tm_loc[1]] = p1[1]
                    dis_hist[tm_loc[0],tm_loc[1]] = p1[2]
                    
                
    return vel_hist,dis_hist

def kinemetry_PA(kine_map,mass_map):
    sh = kine_map.shape
    sym_kmap = np.zeros((sh[0],sh[1]))

    for i in range(sh[0]):
        for j in range(sh[1]):
            tmx = int(2.*(int(sh[0]/2.)-i))
            tmy = int(2.*(int(sh[1]/2.)-j))

            if kine_map[i,j] < -500 or kine_map[i,j+tmy] < -500 or kine_map[i+tmx,j] < -500 or kine_map[i+tmx,j+tmy] < -500:
                tmv = 0.
            else:
                tmv = (kine_map[i,j])+(kine_map[i,j+tmy])-(kine_map[i+tmx,j])-(kine_map[i+tmx,j+tmy])
            sym_kmap[i,j] = tmv/4.

            tmv = (kine_map[i,j])+(kine_map[i+tmx,j])
            
    angs = np.arange(0.,360.,360./150.)
    chi2 = np.zeros(len(angs))
    for i in range(len(angs)):
        ang1 = angs[i]
        tm_vm = spint.rotate(sym_kmap,ang1,reshape=False)
        tm_di = (tm_vm-kine_map)**2.
        chi2[i] = np.sum(tm_di*mass_map**2.)/(np.sum(mass_map**2.))

    tt = np.where(chi2 == np.min(chi2))
    tt = tt[0]

    loc_min = argrelextrema(chi2,np.less)
    loc_min = loc_min[0]
    if len(loc_min) <= 1:
        tang = tt
        fl = 0
    else:
        mchi2s = chi2[loc_min]
        mcord  = sorted(mchi2s)
        t1,t2 = np.where(mchi2s == mcord[0]),np.where(mchi2s == mcord[1])
        tang = [loc_min[t1[0][0]],loc_min[t2[0][0]],int((loc_min[t1[0][0]]+loc_min[t2[0][0]])/2.)]
        fl = 1

    if fl == 0:
        ang_out = angs[tang]
        ang_out = ang_out[0]
        chi_out = chi2[tang]
        chi_out = chi_out[0]
    else:
        ang_out = angs[tang[2]]
        chi_out = chi2[tang[2]]


    return (ang_out)*(np.pi/180.),chi_out

def kPA(kine_map,mass_map):
    km2,mm2 = spint.rotate(kine_map,90.,reshape=False),spint.rotate(mass_map,90.,reshape=False)

    ang1,chi1 = kinemetry_PA(kine_map,mass_map)
    ang2,chi2 = kinemetry_PA(km2,mm2)

    if chi1 < chi2:
        ang_out = ang1
    else:
        ang_out = ang2-(np.pi/2.)

    return ang_out

def re_pixels(si,x0,y0,a,b,pa,r):
    x,y = np.linspace(1.,si,si),np.linspace(1.,si,si)
    xv,yv=np.meshgrid(x,y)

    ellip_grid = (((((xv-x0)*np.cos(pa))-((yv-y0)*np.sin(pa)))**2.)/(a**2.)) + (((((xv-x0)*np.sin(pa))+((yv-y0)*np.cos(pa)))**2.)/(b**2.))

    tt = np.where(ellip_grid > r)
    t2 = np.where(ellip_grid <= r)

    return tt,ellip_grid,t2



# Get the gas mass fraction
def get_gas_frac(ff):

    data = illload_gas(ff)
    stim = (data[:,6] == 1)
    stim = np.where(stim == True)
    stim = stim[0]
    gtim = (data[:,6] == 2)
    gtim = np.where(gtim == True)
    gtim = gtim[0]
    if len(gtim) > 0:

        gfrac = np.sum(data[gtim,8])/(np.sum(data[gtim,8])+np.sum(data[stim,8]))
        sfr   = np.sum(data[gtim,9])
        gtot = np.sum(data[gtim,8])
        stot = np.sum(data[stim,8])

    else:
        sfr = 0.0
        gfrac = 0.0
        gtot = 0.0
        stot = 0.0

    

    return gfrac,gtot*7.3e9,stot*7.3e9

# Measure rotational velocity
def get_vrot(vmap,kpa):
    
    si = float(vmap.shape[0])
    
    x,y = np.linspace(1.,si,int(si)),np.linspace(1.,si,int(si))
    xv,yv=np.meshgrid(x,y)
    rad = np.sqrt((xv-si/2.)**2.+(yv-si/2.)**2.)
    
    m = (-1.)*np.tan(kpa+(np.pi/2.))
    b = (si/2.)-(m*si/2.)

    yl = np.sqrt((m*xv + b - yv)**2.)
    tr,tc = np.where((yl <= 1.5)&(vmap != -1000.))

    R,V = rad[tr,tc],np.abs(vmap[tr,tc])

    tplat = np.where(R >= np.max(R)/2.)
    vplat = np.mean(V[tplat])


    return vplat

    


# Script to generate 50 random viewpoints of subhalo "sim_num" with specified
# pixelscales (kpc/pixel) and fwhm (seeing, kpc). Can also specify an image size
# in kpc, default = 18
def shaper_one(sim_num,pixelscale,fwhm_kpc,size=18,floc='./subhalos/',fout_loc='outputs/',nout=50,profout='mtest1.fits'):
    # Determine image size in pixels
    size       = 18.
    axsi       = round(size/pixelscale)
    while round(axsi/2.) == axsi/2.:
        print(size,axsi)
        size+=0.01
        axsi=round(size/pixelscale)

    # Load the Illustris subhalo
    ff = floc+'/cutout_'+str(sim_num)+'.hdf5'
    data=illload(ff)

    # Determine the median position and velocity then centre
    # galaxy at 0
    tshape = range(len(data[:,0]))
    zp = [np.median(data[tshape,0]),
        np.median(data[tshape,1]),
        np.median(data[tshape,2])]
        
    zv = [np.median(data[tshape,3]),
        np.median(data[tshape,4]),
        np.median(data[tshape,5])]

    xyz=data[tshape,0:3]-zp
    vxyz=data[tshape,3:6]-zv
    mass = data[tshape,9]
    nprts = len(xyz[:,0])

    # Measure the 3D shape
    pp3d,qq,tms,xyz3d,cnt,vxyz3d,omeg = measure_pql(mass,np.array([xyz[:,0],xyz[:,1],xyz[:,2]]),np.array([vxyz[:,0],vxyz[:,1],vxyz[:,2]]))

    # Initialise loop parameters
    nn = nout-1
    p1 = 0
    afact= pi/180.
    prcount=0

    # Open output text file and write in the header info
    fout = open(fout_loc+'Ill'+str(sim_num)+'_50rand_ell_psi.dat','w')
    fout.write('# Illustris '+str(sim_num)+': p = '+str(pp3d)+' ; q = '+str(qq)+'\n#\n')
    fout.write('# pro_ell  pro_ell_err  pro_psi  pro_psi_err  jre  lambda_re  sigma_m  v/sigma vrot\n')

    # loop!
    while p1 <= nn:
        # Rotate data to random viewpoint and create the image (mass or lum. map).
        # This is saved to a fit file to be read by profit (R).
        phi,theta,psi = rand_vp()
        xyz2,vxyz2 = single_rot(xyz[:,0],xyz[:,1],xyz[:,2],vxyz[:,0],vxyz[:,1],vxyz[:,2],phi,theta,psi)
        mass_im = particle_2_image(xyz2,mass,1.,1.,pixelscale,size,fwhm_kpc,file_out=profout)
    
        hdu = pf.open(profout)
        img = hdu[0].data

        xc1,yc1 = float(img.shape[0])/2.,float(img.shape[1])/2.
        
        ngauss = 11
        sigpsf = 0.2
        mvfrac = 0.02
        tt = np.where(img > 0.)
        mv  = np.min(img[tt[0]])+(mvfrac*(np.max(img[tt[0]])-np.min(img[tt[0]])))

        # Attempt to run profit on image, sometimes this fails...
        try:
            pro_out = subp.run(['Rscript','Ill_profit.R',profout],capture_output=True).stdout
            prcount=0
        except:
            pro_out = 'Fail'

            # Counting failures. This can be used to check out why
            # some subhalos fail with profit a lot of the time.
            prcount+=1
            if prcount >= 10:
                p1 = 10000.

        # If profit doesn't fail, continue
        if pro_out != 'Fail':

            # save the outputs of profit
            pro_out = pro_out.split()
            
            el_pro,pa_pro,re_pro,ele_pro,pae_pro = float(pro_out[0]),float(pro_out[1]),float(pro_out[2]),float(pro_out[3]),(np.pi/180.)*float(pro_out[4])
            pa_pro = str(pa_pro)
            pp    = afact*(90+float(pa_pro))
            aa,bb = re_pro,(1.-float(el_pro))*re_pro

            # Initialise good pixel map
            imsi = mass_im.shape
            t_re,rad_im,t_re2 = re_pixels(imsi[0],xc1,yc1,aa,bb,(-1.)*(pp),1.)
            gp_map = rad_im*0.+1.
            gp_map[t_re] = 0.

            # Create the velocity and dispersion maps
            v_map,d_map = particle_2_kmaps(xyz2,vxyz2,mass,gp_map,mass_im,1.,1.,pixelscale,size,fwhm_kpc,70.4,'./vmap.fits',5.e4)

            # Measure lambda_re, j, sigma_m, v/sigma, psi
            lre         = ((np.sum(mass_im[t_re2]*rad_im[t_re2]*(np.abs(v_map[t_re2]))))/(np.sum(mass_im[t_re2]*rad_im[t_re2]*np.sqrt((v_map[t_re2]*v_map[t_re2])+(d_map[t_re2]*d_map[t_re2])))))
            little_jre  = (np.sum(mass_im[t_re2]*rad_im[t_re2]*(np.abs(v_map[t_re2]))))/(np.sum(mass_im[t_re2]))
            sigmam      = (np.sum(mass_im[t_re2]*d_map[t_re2])/np.sum(mass_im[t_re2]))
            vos         = np.sqrt(np.sum(v_map[t_re2]*v_map[t_re2])/np.sum(d_map[t_re2]*d_map[t_re2]))
            
            kine_pa = kPA(v_map,mass_im)
            while kine_pa > 2.*np.pi: kine_pa-=(2.*np.pi)
            pa_pro  = (np.pi/180.)*((-1.)*float(pa_pro))

            vrot = get_vrot(v_map,kine_pa)
                
            psi = np.abs(float(pa_pro)-kine_pa)
            while psi > np.pi/2.: psi = abs(np.pi-psi)

            if p1 == 0:
                vms = np.zeros((nout,v_map.shape[0],v_map.shape[1]))
                dms,mms = np.copy(vms),np.copy(vms)

            vms[p1] = v_map
            dms[p1] = d_map
            mms[p1] = mass_im
            
            p1+=1
            lo = str(float(el_pro))+' '+str(ele_pro)+' '+str(psi)+' '+str(pae_pro)+' '+str(little_jre)+' '+str(lre)+' '+str(sigmam)+' '+str(vos)+' '+str(vrot)+'\n'
            fout.write(lo)
    
    fout.close()


