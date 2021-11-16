import numpy as np
from scipy import io, stats, signal
import numpy.matlib
import scipy.linalg as la
import scipy.sparse.linalg as sla
from numpy.fft import fft, ifft, rfft
import sys
from scipy.interpolate import interp1d
from scipy.signal.windows import dpss
from scipy.stats.distributions import chi2
from scipy.stats import t, binom
from scipy.special import beta
import matplotlib.pyplot as plt
import warnings
import random
import h5py


def filldefault_recursive(opts, key, val):
    optsused = opts
    if ( isinstance(val, dict) ):
        if (key not in opts):
            optsused[key] = dict()
        for jname in val:
            optsused[key] = filldefault_recursive(opts[key], jname, val[jname])
    else:
        if (key not in opts):
            optsused[key] = val;
    return optsused

def checkIfVector(v,shape='N'):
    if v.ndim>2:
        return False, v
    elif v.ndim==2 and not (v.shape[1] == 1 or v.shape[0] == 1):
        return False, v
    else:
        if not shape in ['(1,N)','(N,1)','N']:
            shape == 'N'
        N = np.max(v.shape)
        v.shape = eval(shape)
        return True, v
    
def getfgrid(Fs, nfft, fpass):
    df = Fs/nfft
    f = np.arange(0,np.ceil(Fs/df))*df # all possible frequencies
    f = f[0:int(np.floor(nfft))]
    if len(fpass)!=1:
        func = lambda x: (x >= fpass[0]) and (x<=fpass[-1])
        findx = [i for (i, val) in enumerate(f) if func(val)]
        f=f[findx]
    else:
        findx =np.argmin(np.abs(f-fpass));
        f=[f[findx]]
     
     
    return f, findx

def dpsschk(tapers, N, Fs):
    if isinstance(tapers,list) and len(tapers)==2:
        tapers, eigs= dpss(int(N),tapers[0],Kmax=tapers[1],return_ratios=True)
        tapers = tapers*np.sqrt(Fs)
        return tapers.T, eigs
    elif N!=tapers.shape[1]:
        raise ValueError('''seems to be an error in your dpss calculation; 
                          the number of time points is different from the length 
                          of the tapers''')
    return tapers.T, None

def mtfftc(data, tapers, nfft, Fs):
    isV, data = checkIfVector(data,shape='(N,1)')
    if not isV:
        warnings.warn('data is a matrix, assuming each column is a seperate trial. data.shape = '+str(data.shape))
    NC,C = data.shape # size of data
    NK, K = tapers.shape # size of tapers
    if NK!=NC:
        raise ValueError('length of tapers is incompatible with length of data')
    # add channel indices to tapers
    tapers = np.expand_dims(tapers, axis=2)
    tapers = np.tile(tapers, [1,1,C])
    # add taper indices to data
    data = np.expand_dims(data, axis=2)
    data = np.tile(data, [1,1,K])
    data = np.moveaxis(data, [0,1,2], [0,2,1])


    data_proj = np.multiply(data, tapers)  # product of data with tapers

    J=fft(data_proj,int(nfft),axis=0)/Fs;   # fft of projected data

    return J

def bayesian_confidence_limits(n, h, alpha):
    intervalProbs = binom.pmf(np.arange(0, n+1, 1), n, h)
    sortedIntPrs = np.sort(intervalProbs)[::-1]
    included_dIntProbs = sortedIntPrs[0:np.where( np.cumsum(sortedIntPrs)> 1 - alpha )[0][0] + 1]
    idxs_first = [ np.where( intervalProbs==xi )[0][0] for xi in included_dIntProbs]
    idxs_last = [ np.where( intervalProbs==xi )[0][-1] for xi in included_dIntProbs]
    idxs = np.sort(np.unique(idxs_last + idxs_first))

    if np.size(idxs) == 0:
        lowerIndex = -1
        upperIndex = n
    else:
        lowerIndex = np.min(idxs) - 1
        upperIndex = np.max(idxs)

    cumIntProbs = np.cumsum(intervalProbs)
    if( lowerIndex == -1 ):
        lowerPredictedCoverage = -np.inf
    else:
        lowerPredictedCoverage = cumIntProbs[lowerIndex]

    if( upperIndex == n ):
        upperPredictedCoverage = np.inf
    else:
        upperPredictedCoverage = cumIntProbs[upperIndex]

    return lowerIndex, upperIndex, lowerPredictedCoverage, upperPredictedCoverage

def analytical_scalefactor_Robust(ndof,nsegs,opts = {}):
    opts=filldefault_recursive(opts,'h',0.5)
    opts=filldefault_recursive(opts,'ds',1e-5)
    opts=filldefault_recursive(opts,'n_empiric',0)
    opts=filldefault_recursive(opts,'order_convention','matlab')
    opts=filldefault_recursive(opts,'interp_convention','linear')
    
    f_empiric = np.nan
    f_analytic = 0;

    svals = np.arange(0, 1 + opts['ds'], opts['ds'])
    
    svals = svals[1:-1]

    chiinv_vals = chi2.ppf(svals, df=ndof)

    if ( opts['n_empiric'] > 0 ):
        xsq = np.random.randn(ndof, nsegs, opts['n_empiric']) ** 2
        meansq = np.reshape(np.mean(xsq, axis=0), (nsegs, opts['n_empiric']))
        q = np.quantile(meansq, opts['h'], 1)
        f_empiric = np.mean(q)
    
    N = nsegs - 1
    opts['k_tethered'] = opts['h'] * N
    opts['k_matlab'] = min(max(0, opts['h'] * (N+1) - 1/2), N)
    if( opts['order_convention'] == 'matlab' ):
        opts['k_used'] = opts['k_matlab']
    elif( opts['order_convention'] == 'tethered' ):
        opts['k_used'] = opts['k_tethered']
    k = opts['k_used']
    
    if ( k == np.floor(k) or opts['interp_convention'] == 'beta'):
        f_analytic = (1/beta(k+1, N-k+1)) * opts['ds'] * np.sum((svals ** k) * ((1 - svals) ** (N-k) * chiinv_vals))/ndof
    else:
        k_lo = np.floor(k)
        k_hi = k_lo + 1
        f_lo = (1/beta(k_lo+1, N-k_lo+1)) * opts['ds'] * np.sum((svals ** k_lo) * ((1 - svals) ** (N-k_lo) * chiinv_vals))/ndof
        f_hi = (1/beta(k_hi+1, N-k_hi+1)) * opts['ds'] * np.sum((svals ** k_hi) * ((1 - svals) ** (N-k_hi) * chiinv_vals))/ndof

        f_analytic = (k_hi - k) * f_lo + (k - k_lo) * f_hi
        f_naive = (k_hi - k) * f_lo + (k - k_lo) * f_hi
    opts_used = opts

    return f_analytic,f_empiric,opts_used,f_naive

def scalefactor_Robust(method, dims, types = 'monte-carlo', nruns = 1000):
    nfreqs = dims[0]
    ntapers = dims[1]
    if( len(dims) == 3):
        ntrials = dims[2]
    else:
        ntrials = 1
        
    dims_real = dims
    dims_real[0] = 2
    dims_imag = dims
    dims_imag[0] = 1

    method_standard = {'class':'standard'}
    method_robust = method
    method_robust['scalefactor']['spectrum'] = np.ones((2, 1))
    method_robust['scalefactor']['error'] = np.ones((2, 1))
    if( len(dims_real) < 3 or method['class'] == 'standard' ):
        calib_ratio_s = [1, 1]
        calib_ratio_e = [1, 1]
        types = ''
    elif ( types == 'analytic' and ((method['tier']['estimator'][0] != 'mean') or (method['tier']['estimator'][1] == 'mean'))):
        warnings.warn('Defaulting to monte-carlo scale factor.')
        types = 'monte-carlo'
    
    if ( types != 'monte-carlo' ):
        simruns_nrobust_s = np.zeros((2,nruns), dtype=np.complex_);
        simruns_robust_s  = np.zeros((2,nruns), dtype=np.complex_);
        simruns_nrobust_e = np.zeros((2,nruns), dtype=np.complex_);
        simruns_robust_e  = np.zeros((2,nruns), dtype=np.complex_);
        real_size = tuple(dims_real + [nruns])
 
        imag_size = tuple(dims_imag + [nruns])
   
        simJ = np.vstack((np.zeros(real_size), np.random.normal(0.0, 1.0, size=real_size))) + \
                1j * np.vstack((np.zeros(imag_size), np.random.normal(0.0, 1.0, size=imag_size)))

        simJ = simJ * np.conj(simJ)
        
        for i in range(nruns):
            # TAG
            nrobust_est_s = tapered_estimate_Robust(simJ[:,:,:,i], method_standard, 'spectrum')
            nrobust_est_e = np.squeeze(np.mean(tapered_estimate_Robust(simJ[:,:,:,i], method_standard, 'error'),1))
            robust_est_s  = tapered_estimate_Robust(simJ[:,:,:,i], method_robust,   'spectrum')
            robust_est_e  = tapered_estimate_Robust(simJ[:,:,:,i], method_robust,   'error')

            simruns_nrobust_s[:,i] = nrobust_est_s
            simruns_robust_s[:,i] = robust_est_s
            simruns_nrobust_e[:,i] = nrobust_est_e
            simruns_robust_e[:,i]  = robust_est_e
        
        calib_ratio_s = np.mean(simruns_robust_s, 1) / np.mean(simruns_nrobust_s, 1)
        calib_ratio_e = np.mean(simruns_robust_e, 1) / np.mean(simruns_nrobust_e, 1)
    
    if ( types == 'analytic' ):
        if ( method['class'] == 'one-tier' ):
            if( method['tier']['estimator'][0] == 'median' ):
                h = 0.5
            elif ( method['tier']['estimator'][0] == 'quantile' ):
                h = method['tier']['params'][0]['h']
            ntrials = ntrails * ntapers
            ntapers = 1
        elif ( method['tier']['estimator'][1] == 'median'):
            h = 0.5
        elif ( method['tier']['estimator'][1] == 'quantile'):
            h = method['tier']['params'][1]['h']
    
        s_dc,_,_,_ = analytical_scalefactor_Robust(1*ntapers,   ntrials, {'h':h})
        
        s_el,_,_,_ = analytical_scalefactor_Robust(2*ntapers,   ntrials, {'h':h})
        e_dc,_,_,_ = analytical_scalefactor_Robust(1,   ntapers*ntrials, {'h':h})
        e_el,_,_,_ = analytical_scalefactor_Robust(1, 2*ntapers*ntrials, {'h':h})

        calib_ratio_s = np.array([s_dc, s_el])
        calib_ratio_e = np.array([e_dc, e_el])
    
    indices = np.ones((nfreqs), dtype=np.int)
    
    indices[0] = 0
    indices[-1] = 0
    calib_ratio_s = calib_ratio_s[indices]
    calib_ratio_e = calib_ratio_e[indices]
    
    method_out = method

    method_out['scalefactor']['spectrum'] = calib_ratio_s
    method_out['scalefactor']['error'] = calib_ratio_e
    return method_out

def analytical_scalefactor_Robust(ndof,nsegs,opts = {}):
    opts=filldefault_recursive(opts,'h',0.5)
    opts=filldefault_recursive(opts,'ds',1e-5)
    opts=filldefault_recursive(opts,'n_empiric',0)
    opts=filldefault_recursive(opts,'order_convention','matlab')
    opts=filldefault_recursive(opts,'interp_convention','linear')
    
    f_empiric = np.nan
    f_analytic = 0;

    svals = np.arange(0, 1 + opts['ds'], opts['ds'])

    svals = svals[1:-1]

    chiinv_vals = chi2.ppf(svals, df=ndof)

    if ( opts['n_empiric'] > 0 ):
        xsq = np.random.randn(ndof, nsegs, opts['n_empiric']) ** 2
        meansq = np.reshape(np.mean(xsq, axis=0), (nsegs, opts['n_empiric']))
        q = np.quantile(meansq, opts['h'], 1)
        f_empiric = np.mean(q)
    
    N = nsegs - 1
    opts['k_tethered'] = opts['h'] * N
    opts['k_matlab'] = min(max(0, opts['h'] * (N+1) - 1/2), N)
    if( opts['order_convention'] == 'matlab' ):
        opts['k_used'] = opts['k_matlab']
    elif( opts['order_convention'] == 'tethered' ):
        opts['k_used'] = opts['k_tethered']
    k = opts['k_used']
    
    if ( k == np.floor(k) or opts['interp_convention'] == 'beta'):
        f_analytic = (1/beta(k+1, N-k+1)) * opts['ds'] * np.sum((svals ** k) * ((1 - svals) ** (N-k) * chiinv_vals))/ndof
    else:
        k_lo = np.floor(k)
        k_hi = k_lo + 1
        f_lo = (1/beta(k_lo+1, N-k_lo+1)) * opts['ds'] * np.sum((svals ** k_lo) * ((1 - svals) ** (N-k_lo) * chiinv_vals))/ndof
        f_hi = (1/beta(k_hi+1, N-k_hi+1)) * opts['ds'] * np.sum((svals ** k_hi) * ((1 - svals) ** (N-k_hi) * chiinv_vals))/ndof

        f_analytic = (k_hi - k) * f_lo + (k - k_lo) * f_hi
        f_naive = (k_hi - k) * f_lo + (k - k_lo) * f_hi
    opts_used = opts

    return f_analytic,f_empiric,opts_used,f_naive

def estimate(S, tiermethod, params):
    if( tiermethod == 'mean' ):
        S_est = np.mean(S, axis = 1)
    elif ( tiermethod == 'median' ):
        S_est = np.median(S, axis = 1)
    elif ( tiermethod == 'quantile'):
        h = params['h']
        if (S.ndim == 3):
            S_est = np.transpose( np.quantile(np.transpose(S, (1, 0, 2)), h), (1, 0, 2))
        else:
            S_est = np.quantile(S, h, 1)
    else:
        sys.exit('In calc_dist_midpoint.m: Invalid function input ' +  tiermethod +  '.')
    return np.squeeze(S_est)

def tapered_estimate_Robust(J_est, method, calc_type, trialave = 1):
    nfreqs  = np.shape(J_est)[0];
    ntapers = np.shape(J_est)[1];
    ntrials = np.shape(J_est)[2];
    
    if ( method['class'] == 'standard' ):
        S = np.squeeze(np.mean(J_est, axis=1))
        if (trialave and calc_type == 'spectrum' ):
            S = np.squeeze( np.mean(S, axis = 1) )
            # TAG
    elif ( method['class'] == 'two-tier' ):
        scalefact = method['scalefactor'][calc_type]

        if(calc_type == 'spectrum'):
            S = estimate(J_est, method['tier']['estimator'][0], method['tier']['params'][0])
            if( trialave ):
                S = estimate(S, method['tier']['estimator'][1], method['tier']['params'][1])
                S = S / np.squeeze(scalefact)
            else:
                scalefact = np.tile(scalefact, (1, ntrials))
                S = S / scalefact
        elif (calc_type == 'error'):
            if ( trialave ):
                S = np.reshape(J_est, (nfreqs, ntapers * ntrials))
                S = estimate(S, method['tier']['estimator'][1], method['tier']['params'][1])

                S = S / np.squeeze(scalefact)
            else:
                S = estimate(J_est, method['tier']['estimator'][0], method['tier']['params'][0])
                scalefact = np.tile(scalefact, (1, ntrials))
                S = S / scalefact
                
    elif ( method['class'] == 'one-tier' ):
        scalefact = method['scalefactor'][calc_type]
        if ( trialave ):
            S = np.reshape(J_est, (nfreqs, ntapers*ntrials))
        else:
            S = J_est
            scalefact = np.tile(method['scalefactor'][calc_type], (1, ntrials))
        
        S = estimate(S, method['tier']['estimator'][0], method['tier']['params'][0])
        S = S / scalefact
    return S

def specerr_Robust(S, J, err, trialave, method):
    assert err[0] != 0, 'Need err(1)>0 for error bar calculation. Make sure you are not asking for the output of Serr'
    if( method['class'] == 'two-tier' and 
       (method['tier']['estimator'][1] == 'median' or method['tier']['estimator'][1] == 'quantile')):
         if(err[0] == 1 or err[0] == 2):
                warnings.warn('Asymptotic (err(1)=1) or Jackknife (err(1)=2)) are not recommended for ' + method['tier']['estimator'][1] + '.')
    
    nf, nTapers, nTrials = np.shape(J)
    errtype = err[0]
    p = err[1]
    pp = 1 - p/2
    qq = 1 - pp
    
    if(errtype == 1): # Asymptotic estimate of error
        if (trialave):
            nSamples = nTapers * nTrials
            dof = 2 * nSamples
        else:
            nSamples = nTapers
            nReps = nTrials
            dof = 2 * nSamples * np.ones((1, nReps))
        Qp = chi2.ppf(pp, df=dof)
        Qq = chi2.ppf(qq, df=dof)
        Serr = np.zeros((2, nf, nReps), dtype=np.complex_)
        Serr[0, :, :] = dof.repeat(nf, 0) * S / Qp.repeat(nf, 0)
        Serr[1, :, :] = dof.repeat(nf, 0) * S / Qq.repeat(nf, 0)
        
    elif (errtype == 2): # Jackknife estimate of error
        if (trialave):
            nSamples = nTapers * nTrials
            nReps = 1
            J = np.reshape(J, (nf, nSamples))
        else:
            nSamples = nTapers
            nReps = nTrials
        tcrit = stats.t.ppf(pp, nSamples - 1)
        J = J * np.conj(J)
        Sjk = np.zeros((nSamples, np.shape(J)[0], 1), dtype=np.complex_)
        for k in range(nSamples):
            indices = np.setdiff1d(np.arange(0, nSamples, 1), k)
            Jjk = J[:, indices, np.newaxis]
            eJjk = tapered_estimate_Robust(Jjk, method, 'error', trialave)
            Sjk[k, :, :] = np.reshape(eJjk, (-1, 1))
        sigma = np.sqrt(nSamples - 1) * np.squeeze( np.std(np.log(Sjk), axis=0) )
        sigma = np.reshape(sigma, (-1, 1))
    
        conf = np.tile(tcrit, (nf, nReps)) * sigma;
        Serr = np.zeros((2, nf, nReps), dtype=np.complex_)
        S = np.reshape(S, (-1, 1))
        Serr[0, :, :] = S * np.exp(-conf)
        Serr[1, :, :] = S * np.exp(conf)
        
    elif (errtype == 3):  # Bayesian confidence limits, for use with median or quantile
        J_est = np.conj(J) * J
        if( method['class'] == 'two-tier' ):
            nSamples = nTrials
            if ( trialave ):
                trial_est = method['tier']['estimator'][1]
                if( method['tier']['estimator'][0] == 'mean'):
                    J_est = np.squeeze(np.mean(J_est, axis = 1))
                elif( method['tier']['estimator'][0] == 'median'):
                    J_est = np.squeeze(np.median(J_est, axis = 1))
                elif( method['tier']['estimator'][0] == 'quantile'):
                    h = method['tier']['params'][0]['h']
                    if( J_est.ndim == 3):
                        J_est = np.transpose( np.quantile(np.transpose(J_est, (1, 0, 2)), h), (1, 0, 2))
                    else:
                        J_est = np.quantile(S, h, 1)
        elif ( method['class'] == 'one-tier' ):
            nSamples = nTapers * nTrials
            trial_est = method['tier']['estimator'][0]
            if ( trialave ):
                J_est = np.squeeze( reshape(J_est, (nf, nSamples)) )
        elif ( method['class' == 'standard'] ):
            sys.exit('Standard method cannot be implemented with Bayesian error bars.')
        
        if (trial_est == 'quantile' ):
            h = method['tier']["params"][1]['h']
        elif (trial_est == 'median' ):
            h = 0.5
        elif (trial_est== 'mean' ):
            sys.exit('Mean estimator cannot be implemented with Bayesian error bars.')
        
        Serr = np.zeros((2, nf, 1), dtype=np.complex_)
        n = nSamples
        sortedData = np.sort(J_est, 1)
        lowIndex, highIndex, _, _ = bayesian_confidence_limits(n, h, p);
        if (lowIndex < 1):
            warnings.warn('Not enough data samples to calculate finite Bayesian error bars for a p-value.')
            Serr[0, :, :] = -np.inf
            Serr[1, :, :] = np.inf
        else:
            scalefact = method['scalefactor']['spectrum']
            if( trialave ):
                Serr[0, :, :] = (sortedData[:, lowIndex] / scalefact).T.reshape((-1, 1))
                Serr[1, :, :] = (sortedData[:, highIndex]/ scalefact).T.reshape((-1, 1)) 
            else:
                Serr[0, :, :] = sortedData[:, lowIndex].reshape((-1, 1)).T / np.tile(scalefact, (1, ntrials))
                Serr[1, :, :] = sortedData[:, highIndex].reshape((-1, 1)).T / np.tile(scalefact, (1, ntrials))
    elif (errtype == 4): # Bootstrap error estimate
        if (trialave):
            nSamples = nTapers * nTrials
            nReps = 1
        else:
            nSamples = nTapers
            nReps = nTrials
        
        nboot = 10000;
        Jboot = np.zeros((nf, nboot, nReps), dtype=np.complex_)
        Serr = np.zeros((2, nf, nReps), dtype=np.complex_)
        J_est = J * np.conj(J)
        for b in range(nboot):
            indices = np.ceil(np.random.uniform(0, nTrials, (nTrials, 1)))
            Jboot[:, b, :] = tapered_estimate_Robust(J_est[:, :, indices], method, 'spectrum', trialave)
        Jboot = np.sort(Jboot, 1)
        lowerLimIndex = np.floor(nboot * qq)
        upperLimIndex = np.ceil(nboot * pp)
        lowerlimVals = Jboot[:, upperLimIndex, :]
        upperLimVals = Jboot[:, lowerLimIndex, :]
        
        Serr[0, :, :] = lowerlimVals
        Serr[1, :, :] = upperLimVals
        
    return np.squeeze(Serr)


def getparams(params):
    if( 'tapers' in params):
        tapers = params['tapers']
    else:
        tapers = [3, 5]
        
    if ( 'pad' in params):
        pad = params['pad']
    else:
        pad = 0
        
    if( 'Fs' in params ):
        Fs = params['Fs']
    else:
        Fs = 1
        
    if ( 'fpass' in params):
        fpass = params['fpass']
    else:
        fpass = [0, Fs/2]
    
    if ( 'err' in params):
        err = params['err']
    else:
        err = 0
        
    if ( 'trialave' in params):
        trialave = params['trialave']
    else:
        trialave = 0
        
    return tapers, pad, Fs, fpass, err, trialave, params

def mtspectrumc_Robust(*args):
    if(len(args) < 1):
        sys.exit('Need data')
    elif (len(args) < 2):
        params = {'trialave':1}
        data = args
    elif (len(args) < 3):
        data, params = args
        print('Defaulting to robust estimation method.')
        method = {
            'class':'two-tier',
            'tier':{
                'estimator':['mean', 'median']
            }
        }
    elif (len(args) < 4):
        data, params, method = args
        if(method['class'] == 'two-tier' and method['tier']['estimator'][0] == 'mean' and 
          (method['tier']['estimator'][1] == 'mean' or method['tier']['estimator'][1] == 'quantile') ):
            scalefactor_calculation = 'analytic'
        else:
            scalefactor_calculation = 'monte-carlo'
    else:
        data, params, method, scalefactor_calculation = args
    # TAG
#     scalefactor_calculation = 'monte-carlo'
    if( 'trialave' not in params ):
        params['trialave'] = 1
        
    tapers, pad, Fs, fpass, err, trialave, params = getparams(params);
    if ( err[0] == 0 ):
        sys.exit('When Serr is desired, err(1) has to be non-zero.')
        
    N = np.shape(data)[0]
    # 750 20 1
    nfft = int(max( 2**( np.ceil(np.log2(abs(N))) + pad ), N))
    f, findx = getfgrid(Fs, nfft, fpass);
    tapers, _ = dpsschk(tapers, N, Fs); # check tapers
    J = mtfftc(data,tapers, nfft, Fs);
    
    J = J[findx, :, :]

    if ( method['class'] != 'standard' and ( ('scalefactor' not in method ) or len(method['scalefactor']['spectrum']) == 0) ):
        method['scalefactor'] = {}
        method = scalefactor_Robust(method, list(np.shape(J)), scalefactor_calculation)
    elif ( len(args) == 4 ):
        warnings.warn('S calefactor is precalculated but scalefactor_calculation is passed in; defaulting to precalculated values.')
    
    if ( trialave == 0 and method['class'] == 'standard' ):
        warnings.warn('Non-standard method should only be implemented if trialave==1.')
                      
    J_est = J * np.conj(J)
    S = tapered_estimate_Robust(J_est, method, 'spectrum', trialave)
    Serr = specerr_Robust(S, J, err, trialave, method)

    
    
    return S, f, Serr, method

"""
    data: input data, shape: (nPnts, nTrials)
    type: estimation type, value='robust'/'standard'
"""

def computePowerEstimation(data, type='robust', NW=3, Kmax=5, Fs=250.0, err0=2, quantile_h = 0.5):
    quantile_h = 0.5
    
    mts_args = {'trialave':1, 'tapers':[NW, Kmax], 'err':[err0, 0.05], 'Fs': Fs}
    standard_method = {'class':'standard'}
    robust_method = {
        'class':'two-tier', 
        'tier':{
            'estimator':['mean', 'quantile'],
            'params':[
                {},
                {'h':quantile_h}
            ]
        }
    }
    
    if(type == 'robust'):
        S, f, Serr, _ = mtspectrumc_Robust(data, mts_args, robust_method)
    else:
        S, f, Serr, _ = mtspectrumc_Robust(data, mts_args, standard_method)
    
    return S, f, Serr