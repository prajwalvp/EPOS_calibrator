#!/usr/bin/python

import psrchive
import commands
import numpy as np
import sys
import math
import EPOS_calibrator
import pa_calculator


def archive_to_channels_normalized(filename):
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
    arch = psrchive.Archive_load(filename)
    arch.remove_baseline()
    arch.tscrunch()
    data = arch.get_data()
    data = data.squeeze()

    return np.vstack((data[0],data[1],data[2],data[3]))

def off_pulse_rms_I(profile):
    mod_profile = np.roll(profile,500)
    #plt.plot(mod_profile)
    #plt.show()
    noise = mod_profile[150:850]
    index_sets = [np.argwhere(i==noise) for i in np.unique(noise)]
    for i in index_sets:
        if len(i) > 1:
            indices= i[:,0]
            mod_noise = np.delete(noise,indices)
        else:
            mod_noise=noise
    #mad_val = 1.4826*median_absolute_deviation(noise)
    rms_value = np.sqrt(((mod_noise)**2).mean())
    #print mad_val,rms_value
    return rms_value


def off_pulse_median_L(L):
    L_1000 = np.roll(L,1000-np.argmax(L))
    noise = L_1000[150:800]
    L_median = np.median(noise)
    return L_median

def off_pulse_median_I(I):
    I_1000 = np.roll(L,1000-np.argmax(I))
    noise = I_1000[150:800]
    I_median = np.median(noise)
    return I_median


def convert_to_Stokes_params(filename):
    """
    Conversion applied assuming an adding polarimeter
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
 
def mjd(filename):
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
    PA = 0.5*np.degrees(np.arctan2(U,Q))
        
    return PA

def cal_to_gains(cal1,cal2,cal3,cal4,sig3,sig4):
    g1 = cal1
    g2 = cal2
    g3 = 0.5*cal4*(sig3/sig4)
    g4 = 0.5*cal4
    alpha = 1 - (cal3/cal4)*(sig4/sig3)
    return g1,g2,g3,g4,alpha

def channels_to_gain_calibrated_stokes(channels,g1,g2,g3,g4,alpha):
    I = (1/g1)*channels[0] + (1/g2)*channels[1]
    V = (1/g1)*channels[0] - (1/g2)*channels[1]
    Q = (1/alpha)*(I-(1/g3)*channels[2])
    U = (1/alpha)*((1/g4)*channels[3]-I)
    L = np.sqrt(Q**2 + U**2)

    return I,Q,U,V,L


#def remove_cross_coupling()




if __name__=="__main__":
    import matplotlib.pyplot as plt   
    obs_files = np.loadtxt('obs_info.txt',dtype=str)    
    cal_files = np.loadtxt('cal_info.txt',dtype=str)    
    
    for i in range(len(obs_files)):
        orig_file = obs_files[i]
        profile = archive_to_channels_normalized(orig_file)
        channels = archive_to_channels(orig_file)
        cal_file = cal_files[i]
        I,Q,U,V,L = convert_to_Stokes_params(orig_file)

        cal1,cal2,cal3,cal4,sig3,sig4 = EPOS_calibrator.get_cals(cal_file)

        g1,g2,g3,g4,alpha = cal_to_gains(cal1,cal2,cal3,cal4,sig3,sig4)
        I_cal,Q_cal,U_cal,V_cal,L_cal = channels_to_gain_calibrated_stokes(channels,g1,g2,g3,g4,alpha)
            
        SNR = snr(orig_file)
        MJD = mjd(orig_file)
        l = length(orig_file)
        subs = nsubint(orig_file)
        pa = pa_calculator.parallactic_angle(orig_file)

        L_modified = L_cal - off_pulse_median_L(L_cal)
        I_modified = I_cal - off_pulse_median_I(I_cal)
        PA = position_angle(Q,U) 

        I_cal_norm = np.divide(I_cal,np.amax(I_cal))
        L_cal_norm = np.divide(L_cal,np.amax(I_cal))
        V_cal_norm = np.divide(V_cal,np.amax(I_cal))

        I_norm = np.divide(I,np.amax(I))
        L_norm = np.divide(L,np.amax(I))
        V_norm = np.divide(V,np.amax(I))

        plt.figure(1)
        plt.plot(I_cal_norm,label='Stokes I')
        plt.plot(I_norm,label='Stokes I(uncal)')
        plt.title(orig_file+' '+cal_file)
        plt.legend()
        plt.savefig(orig_file+'_uncal_vs_cal.png')
        plt.clf()
        
        plt.figure(2)
        plt.plot(I_norm,label='Stokes I')
        plt.plot(L_norm,label='Stokes L')
        plt.plot(V_norm,label='Stokes V')
        plt.title(orig_file+' '+cal_file)
        plt.legend()
        plt.savefig(orig_file+'_uncalibrated.png')
        plt.clf()

        plt.figure(3)
        plt.plot(I_cal_norm,label='Stokes I(cal)')
        plt.plot(L_cal_norm,label='Stokes L(cal)')
        plt.plot(V_cal_norm,label='Stokes V(cal)')
        plt.title(orig_file+' '+cal_file)
        plt.legend()
        plt.savefig(orig_file+'_calibrated.png')
        plt.clf()

