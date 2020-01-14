#!/usr/bin/python

import psrchive
import commands
import numpy as np
import sys
import math
import EPOS_calibrator
from astropy.stats import median_absolute_deviation

def archive_to_channels_normalized(filename):
    """
    Retrieve normalized Stokes I profile from archive
    """
    arch = psrchive.Archive_load(filename)
    arch.remove_baseline()
    arch.tscrunch()
    data = arch.get_data()
    data = data.squeeze()
    channels=[]

    for i in range(4):
        profile = data[i,:]
        normalized_profile = np.divide(profile,np.amax(profile))
        normalized_profile_500 = np.roll(normalized_profile,500-np.argmax(normalized_profile))
        #Align bin 500 to left pulse no matter what
        if normalized_profile_500[460] < 0.5:
            channels.append(normalized_profile_500)
        else:
            channels.append(np.roll(normalized_profile_500,35))

    return channels



def archive_to_channels(filename):
    """
    Retrieve the 4 channel data as 4 arrays from archive
    """
    arch = psrchive.Archive_load(filename)
    arch.remove_baseline()
    arch.tscrunch()
    data = arch.get_data()
    data = data.squeeze()

    return np.vstack((data[0],data[1],data[2],data[3]))

def off_pulse_rms_I(profile):
    """
    Calculate off pulse r.m.s for Stokes I
    """
    mod_profile = np.roll(profile,500) # Shift profile such that on-pulse region is at the end of rotation phase
    noise = mod_profile[150:850] # Choose window of off-pulse region
    index_sets = [np.argwhere(i==noise) for i in np.unique(noise)]
    for i in index_sets:
        if len(i) > 1:
            indices= i[:,0]
            mod_noise = np.delete(noise,indices)
        else:
            mod_noise=noise
    rms_value = np.sqrt(((mod_noise)**2).mean())
    return rms_value


def off_pulse_median_L(L):
    """
    Calculate off pulse median for Stokes L
    """
    L_1000 = np.roll(L,1000-np.argmax(L))
    noise = L_1000[150:800]
    L_median = np.median(noise)
    return L_median

def off_pulse_median_I(I):
    """
    Calculate off pulse median for Stokes I
    """
    I_1000 = np.roll(L,1000-np.argmax(I))
    noise = I_1000[150:800]
    I_median = np.median(noise)
    return I_median


def convert_to_Stokes_params(filename):
    """
    Conversion applied assuming an adding polarimeter. Equations from Xilouris & Hoensbroch 1996
    """
    arch = psrchive.Archive_load(filename)
    arch.remove_baseline()
    arch.tscrunch()
    data = arch.get_data()
    data = data.squeeze()
    
    I  = data[0] + data[1] # I = Channel1 + Channel2
    Q  = I  - data[2] # Q = I - Channel3
    U  = data[3] - I # U = Channel4 - I
    V = data[0]-data[1] #V = Channel1 - Channel2

    L = np.sqrt(Q**2 + U**2)

    return I,Q,U,V,L
 
def mod_jul_day(filename):
    arch = psrchive.Archive_load(filename)
    mjd_value = float(arch.get_Integration(0).get_start_time().in_days())
  
    return mjd_value

def snr(filename):
    data1 = commands.getoutput("psrstat -c snr -j FTp %s"%filename)
    snr_value = float(data1.split("snr=")[1])
    return snr_value
            
def length(filename):
    data1 = commands.getoutput("psrstat -c length -j FTp %s"%filename)
    length_value = float(data1.split("length=")[1])
    return length_value

def nsubint(filename):
    data1 = commands.getoutput("psrstat -c nsubint  %s"%filename)
    nsubint_value = int(data1.split("nsubint=")[1])
    return nsubint_value

def position_angle(Q,U):
    PA = -np.degrees(np.arctan2(U,Q))
        
    return PA

def cal_to_gains(cal1,cal2,cal3,cal4,sig3,sig4):
    g1 = cal1
    g2 = cal2
    g3 = 0.5*cal4*(sig3/sig4)
    g4 = 0.5*cal4
    alpha = 1 - (cal3/cal4)*(sig4/sig3)
    return g1,g2,g3,g4,alpha

def channels_to_gain_calibrated_stokes(channels,g1,g2,g3,g4,alpha):
    """
    Convert to gain calibrated Stokes Parameters
    """
    I = (1/g1)*channels[0] + (1/g2)*channels[1]
    V = (1/g1)*channels[0] - (1/g2)*channels[1]
    Q = (1/alpha)*(I-(1/g3)*channels[2])
    U = (1/alpha)*((1/g4)*channels[3]-I)
    L = np.sqrt(Q**2 + U**2)

    return I


def first_peak_coord(sampled_combo,val,window_fp):
    y = sampled_combo
    x = np.arange(val-window_fp,val+window_fp+1,1)
    fit,cov = np.polyfit(x,y,2,cov=True)
    uc = np.sqrt(np.diag(cov))
    a = fit[0]
    b = fit[1]
    c = fit[2]
    tmp = -b / (2*a)
    solved_x = tmp
    solved_y = a*(solved_x)**2 + b*(solved_x) + c

    return solved_x,solved_y



def off_pulse_rms(profile):
    mod_profile = np.roll(profile,500)
    noise = mod_profile[150:850]
    index_sets = [np.argwhere(i==noise) for i in np.unique(noise)]
    for i in index_sets:
        if len(i) > 1:
            indices= i[:,0]
            mod_noise = np.delete(noise,indices)
        else:
            mod_noise=noise
    mad_val = 1.4826*median_absolute_deviation(noise)
    rms_value = np.sqrt(((mod_noise)**2).mean())
    #print mad_val,rms_value
    return rms_value

def perform_mc(profile,rms,val,win,no_of_iterations):
    ar = []
    for i in range(val-win,val+win+1):
        ar.append(np.random.normal(profile[i],rms,no_of_iterations))
    ar1 = np.asarray(ar)
    sx=[]
    sy=[]
    for i in range(no_of_iterations):
        fp_y1 = ar1[:,i]
        solved_x,solved_y = first_peak_coord(fp_y1,val,win)
        sx.append(solved_x)
        sy.append(solved_y)

    sx= np.asarray(sx)
    sy= np.asarray(sy)
    sx = np.asarray(sx)
    weights = np.ones_like(sx)/float(len(sx))
    n,bins,rectangles = plt.hist(sx, weights=weights,bins=100)
    cumulative = np.cumsum(n)

    # 5 and 95 percentile
    idx_5 = (np.abs(cumulative - 0.05)).argmin()
    idx_95 = (np.abs(cumulative - 0.95)).argmin()

    median1 = np.median(bins)

    a = median1-bins[idx_5]
    b = bins[idx_95]-median1

    return bins[idx_5],median1,bins[idx_95]


def Ical_to_normalized_profile(I_cal):
    """
    convert calibrated Stokes I   
    """
    profile = I_cal
    normalized_profile = np.divide(profile,np.amax(profile))
    normalized_profile_500 = np.roll(normalized_profile,500-np.argmax(normalized_profile))
    #Align bin 500 to left pulse no matter what
    if normalized_profile_500[460] < 0.5:
        return normalized_profile_500
    else:
        return np.roll(normalized_profile_500,35)



if __name__=="__main__":
    import matplotlib.pyplot as plt  
    obs_files = np.loadtxt('obs_info.txt',dtype=str) # List of observation files
    cal_files = np.loadtxt('cal_info.txt',dtype=str) # List of corresponding calibrator files
 
    for i in range(len(obs_files)):
        orig_file = obs_files[i]
        print orig_file 
        profile = archive_to_channels_normalized(orig_file)
        channels = archive_to_channels(orig_file)
        cal_file = cal_files[i]
        print cal_file
        cal1,cal2,cal3,cal4,sig3,sig4 = EPOS_calibrator.get_cals(cal_file)

        g1,g2,g3,g4,alpha = cal_to_gains(cal1,cal2,cal3,cal4,sig3,sig4)
        I_cal = channels_to_gain_calibrated_stokes(channels,g1,g2,g3,g4,alpha)
        
        f = open('I_cal.txt','a')
        np.savetxt(f,np.c_[I_cal])
        f.close()
