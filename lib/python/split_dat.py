import sys, copy
import subprocess
from infodata import *

NPART = 2


def copy_small_dat(filename, outfilename, npts, ipart):

    # subprocess.DEVNULL is only suppoted in python >3.3
    out_fd = open('dd-log','w')
    
    cmd = "dd if=%s of=%s bs=%d count=1"%(filename, outfilename, npts*4)
    if ipart:
        cmd += " skip=%d"%ipart
    
    #print cmd
    retcode = subprocess.call(cmd, shell=True, stdout=out_fd, stderr=out_fd)
    #retcode = subprocess.call(cmd)


def split_dat(dat_fn, npart):
    """
    Split a .dat file into npart. Create the appropriate .inf file
    """

    outfn = []

    inf_fn = dat_fn.replace(".dat",".inf")

    # Read the .inf structure
    info = infodata(inf_fn)

    # Calculate new number of points
    new_N = info.N / NPART
    # be careful if it's an odd number
    if new_N%2 != 0:
	new_N += 1
	
    for i in range(npart):
	# Copy object
	new_info = copy.copy(info)
	new_info.onoff = []

	# Update the number of points 
	if i == npart-1:
	    new_info.N = info.N - i*new_N
	else:
	    new_info.N = new_N

	# Update the epoch
	new_info.epoch = info.epoch + (i * info.dt * new_info.N) / 86400.

	# Manage breaks in the data
	if info.breaks:
	    #print (i+1)*new_info.N-1, info.onoff[0][1]
	    if (i+1)*new_info.N-1 <= info.onoff[0][1]:
		new_info.breaks = 0
	    else:
		new_info.breaks = 1
		new_info.onoff.append((0, info.onoff[0][1] - new_info.N))
		new_info.onoff.append((info.onoff[1][0] - new_info.N, info.onoff[1][1] - new_info.N))

	# Determine new filename
	new_dat_fn = dat_fn.replace('_DM','-part%dx%d_DM'%(i,npart))

	# Write the .dat file
	copy_small_dat(dat_fn, new_dat_fn, new_N, i)

	# Write the .inf file
	new_info.write_inf(inf_fn.replace('_DM','-part%dx%d_DM'%(i,npart)))	    

	outfn.append(new_dat_fn)

    return outfn

if __name__ == '__main__':
    split_dat(sys.argv[1], NPART)

