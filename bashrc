module use /home/app/modulefiles

module load openblas/cpu-0.3.6-gcc-4.8.5  
module load hdf5/cpu-1.10.4 gcc/cpu-7.3.0
#cuda 10.0
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda-10.0/include:$CPATH


export PATH=/home/blao/anaconda3/bin:$PATH
export CPATH=/home/blao/anaconda3/include/python3.7m:$CPATH
export CPATH=/home/blao/anaconda3/lib/python3.7/site-packages/numpy/core/include:$CPATH


export HDF5_USE_FILE_LOCKING="FALSE"
