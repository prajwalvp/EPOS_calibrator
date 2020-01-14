import psrchive
import sys
import numpy as np


def get_cals(filename):
    """
    Estimate the noise diode calibrator power and it's standard deviation
    """
    arch = psrchive.Archive_load(filename)
    arch.remove_baseline()
    arch.tscrunch()
    data = arch.get_data()
    data = data.squeeze()
    grads = np.gradient(data[0])
    
    cal_bp = [index for index,value in enumerate(grads) if value < -50.0]
    
    cal_bp_true = [i for i in cal_bp if i!=0 and i!=1023]  
    
    cal1 = np.median(data[0][9:cal_bp_true[0]-5])    
    cal2 = np.median(data[1][9:cal_bp_true[0]-5])    
    cal3 = np.median(data[2][9:cal_bp_true[0]-5])    
    cal4 = np.median(data[3][9:cal_bp_true[0]-5])    

    sig3 = np.std(data[2][cal_bp_true[-1]+20:1013])
    sig4 = np.std(data[3][cal_bp_true[-1]+20:1013])

    return cal1,cal2,cal3,cal4,sig3,sig4


def cal_to_gains(cal1,cal2,cal3,cal4,sig3,sig4):
    g1 = cal1
    g2 = cal2
    g3 = 0.5*cal4*(sig3/sig4)
    g4 = 0.5*cal4
    alpha = 1 - (cal3/cal4)*(sig4/sig3)
    return g1,g2,g3,g4,alpha



if __name__=='__main__':
    cal1,cal2,cal3,cal4,sig3,sig4 = get_cals(sys.argv[1])
    g1,g2,g3,g4,alpha = cal_to_gains(cal1,cal2,cal3,cal4,sig3,sig4)

    print g1,g2
