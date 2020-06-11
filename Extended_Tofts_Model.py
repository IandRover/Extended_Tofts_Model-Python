from scipy import integrate 
from scipy.optimize import curve_fit

def Extended_Tofts_Integral(t, Cp, Kt=0.1, ve=0.2, vp=0.1, uniform_sampling=True):
    nt = len(t)
    Ct = np.zeros(nt)
    for k in range(nt):
        tmp = vp*Cp[:k+1] + integrate.cumtrapz(np.exp(-Kt*(t[k]-t[:k+1])/ve)*Cp[:k+1],t[:k+1], initial=0.0)
        Ct[k] = tmp[-1]
    return Ct

def FIT(Ct, Cp, time):
    fit_func = lambda t, Kt, ve, vp: ext_tofts_integral(t, Cp, Kt=Kt, ve=ve,vp=vp)
    ini = [0.01, 0.01, 0.01]
    popt, pcov = curve_fit(fit_func, time, Ct, p0=ini)
    return popt, pcov

