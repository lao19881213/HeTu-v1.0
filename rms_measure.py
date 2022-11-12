#!/usr/bin/env python

import sys
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
from numpy import std, average, min, max

def calculate_rms_from_fits(corners=True, middle=True, mean=False, \
            maximum=False, minimum=False, boxsize=300, filename=None):
    if filename is None:
       print ("Must supply a filename")
       sys.exit(1)
    else:
       side=boxsize
       rms=[]
       hdulist = pyfits.open(filename)
       if hdulist[0].data.shape[0]>1:
          #format is e.g. RA, Dec, stokes, spectral -- unusual for MWA images
          totlen=hdulist[0].data.shape[0]
          sfxy=False
       else:
          # format is e.g. stokes, spectral, RA, Dec -- usual for MWA images
          totlen=hdulist[0].data.shape[2]
          sfxy=True

       scidata=[]
       if totlen>0:
          if corners:
              if sfxy:
                  scidata.append(hdulist[0].data[:,:,totlen-side:totlen,totlen-side:totlen][0][0])
                  scidata.append(hdulist[0].data[:,:,0:side,totlen-side:totlen][0][0])
                  scidata.append(hdulist[0].data[:,:,totlen-side:totlen,0:side][0][0])
                  scidata.append(hdulist[0].data[:,:,0:side,0:side][0][0])
              else:
                  scidata.append(hdulist[0].data[totlen-side:totlen,totlen-side:totlen][0][0])
                  scidata.append(hdulist[0].data[0:side,totlen-side:totlen][0][0])
                  scidata.append(hdulist[0].data[totlen-side:totlen,0:side][0][0])
                  scidata.append(hdulist[0].data[0:side,0:side][0][0])
          else:
              half=int(totlen/2)
              if sfxy:
                  scidata=hdulist[0].data[:,:,half-side:half+side,half-side:half+side]
              else:
                  scidata=hdulist[0].data[half-side:half+side,half-side:half+side]
          if mean:
              return average(scidata)
          elif maximum:
              return max(scidata)
          elif minimum:
              return min(scidata)
          else:
              return std(scidata)
