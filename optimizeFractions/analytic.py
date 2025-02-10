import numpy as np
import pandas as pd


from scipy.stats import norm
from scipy.io import loadmat

from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.stats import norm
import multiprocessing as mp
from scipy.fft import fft, ifft, fftshift,ifftshift
from scipy.signal import fftconvolve, butter, sosfilt

from scipy.stats import rv_histogram

from math import gamma

from scipy.optimize import curve_fit

import quantities as pq

import sys
import time

def getAreaScaleFactor():
    
    ### Overall fiber counts in the nerve
    maffcount = 34000
    meffcount = 14800
    ueffcount = 21800 
    uaffcount = 315000
    ####
    
    #### Loads diameter distributions
    maffvals = np.loadtxt('../Data/maffvals.csv',delimiter=',')
    meffvals = np.loadtxt('../Data/meffvalsSmooth.csv',delimiter=',')
    uaffvals = np.loadtxt('../Data/uaffvals.csv',delimiter=',')
    ueffvals = np.loadtxt('../Data/ueffvals.csv',delimiter=',')
    #####
    
    #### Gets midpoints of histogram bins
    maffD = (maffvals[:-1,0] + maffvals[1:,0])/2 * 1e-6
    maffP = (maffvals[:-1,1] + maffvals[1:,1])/2 

    meffD = (meffvals[:-1,0] + meffvals[1:,0])/2 * 1e-6
    meffP = (meffvals[:-1,1] + meffvals[1:,1])/2

    uaffD = (uaffvals[:-1,0] + uaffvals[1:,0])/2 * 1e-6
    uaffP = (uaffvals[:-1,1] + uaffvals[1:,1])/2

    ueffD = (ueffvals[:-1,0] + ueffvals[1:,0])/2 * 1e-6
    ueffP = (ueffvals[:-1,1] + ueffvals[1:,1])/2
    #######
    
    
    maffArea = np.sum(maffD**2*maffP*maffcount / 100)
    meffArea = np.sum(meffD**2*meffP*meffcount / 100)
    uaffArea = np.sum(uaffD**2*uaffP*uaffcount / 100)
    ueffArea = np.sum(ueffD**2*ueffP*ueffcount / 100)

    totalFiberArea = maffArea + meffArea + uaffArea + ueffArea

    fascicleSizes = np.array([.24*.26,.16*.16,.18*.2,.16*.16,.12*.14,.16*.16,.1*.12,.24*.2,.2*.24,.18*.2,.14*.12,
                    .16*.16,.1*.08,.16*.14,.12*.12,.08*.08,.14*.12,.1*.1,.2*.18,.14*.14,.14*.12,
                .12*.12,.22*.18,.14*.14,.14*.12,.18*.18,.16*.16,.1*.16,.12*.12,.22*.22,.1*.1,.1*.08,
                .12*.12,.1*.1,.12*.1,.14*.1,.1*.1,.14*.12,.18*.16])*1e-3**2
    
    diamScaleFactor = (np.sum(fascicleSizes)/totalFiberArea)
    
    return diamScaleFactor

def getNumFibers(fascicleArea,diamScaleFactor,fascIdx,fascTypes,distribution_params):

    
    #### Assigns fiber counts per fascicle for each type
    
    maffFrac, meffFrac, ueffFrac, uaffFrac = getFiberTypeFractions(fascIdx,fascTypes,distribution_params)
        
    ###########
    
    ### Overall counts in the nerve
    maffcount = 34000
    meffcount = 14800
    ueffcount = 21800 
    uaffcount = 315000
    #######
    
    
    ##### Loads diameter distribution histograms
    maffvals = np.loadtxt('../Data/maffvals.csv',delimiter=',')
    meffvals = np.loadtxt('../Data/meffvalsSmooth.csv',delimiter=',')

    uaffvals = np.loadtxt('../Data/uaffvals.csv',delimiter=',')
    ueffvals = np.loadtxt('../Data/ueffvals.csv',delimiter=',')
    #######
    
    ### Gets midpoints of histogram bins
    maffD = (maffvals[:-1,0] + maffvals[1:,0])/2 *1e-6
    maffP = (maffvals[:-1,1] + maffvals[1:,1])/2 

    meffD = (meffvals[:-1,0] + meffvals[1:,0])/2 * 1e-6
    meffP = (meffvals[:-1,1] + meffvals[1:,1])/2

    uaffD = (uaffvals[:-1,0] + uaffvals[1:,0])/2 * 1e-6
    uaffP = (uaffvals[:-1,1] + uaffvals[1:,1])/2

    ueffD = (ueffvals[:-1,0] + ueffvals[1:,0])/2 * 1e-6
    ueffP = (ueffvals[:-1,1] + ueffvals[1:,1])/2
    #########

 
    maffArea = maffFrac * np.sum(maffD**2*maffP / 100)
    meffArea = meffFrac * np.sum(meffD**2*meffP / 100)
    uaffArea = uaffFrac * np.sum(uaffD**2*uaffP / 100)
    ueffArea = ueffFrac * np.sum(ueffD**2*ueffP / 100)

    fascicleNumber = fascicleArea[fascIdx] / (diamScaleFactor * (maffArea + meffArea + uaffArea + ueffArea)) 
    
#    fascicleNumber = 10000
    
    return maffFrac, meffFrac, uaffFrac, ueffFrac, fascicleNumber


def sampleFractionHistogram(ColorX,ColorY,Colors,side,rng):
    
    ##### In this function, we generate a histogram of per-fascicle fiber type fractions from the paper, and sample from it to produce the fraction for the simulated fascicle  
    
    Interp = interp1d(ColorX,ColorY,bounds_error=False,fill_value = 'extrapolate') # Interpolation object for the color scale bar
    FracList = Interp(Colors[side]) # Interpolates scale bar at the inttensity values for each fascicle in the paper
    FracList[np.where(FracList<0)] = 0
    
    hist = rv_histogram(np.histogram(FracList))
    frac = hist.rvs(size=1,random_state=rng)
    
    return frac*.01 # Converts from percentage to fraction

def getFiberTypeFractions(fascIdx, fascTypes, distribution_params):
    
    
    ###### In this block, we take the color intensity values from the plot of fiber type concentration per fascicle, and the scale bars. 
    
    maffColors = [[103,211,191,255,254,191,157,254,231,232,255,248,255,226,244,255,227,212,192, 255],[160,80,82,81,83,82,118,158,135,182,184,182,134,110,102,128,82,107,119,114,135,126]] # Color intensities for each fascicle. First array corresponds to the left half of the nerve, second array to the right half of the nerve
    
    maffColorY = [0,10,15,20] # Fiber composition percentage from scale bar
    maffColorX = [255,205,151,95] # Corresponding color intensities for these composition percentages
    
    meffColors = [[243,253,180,169,226,202,169,214,198,207,219,177,180,210,226,171,169,169,169],[254,254,254,253,251,254,252,252,254,255,254,254,243,255,255,254,255,255,255,252,254]]
    
    meffColorY = [0,5,10,15]  # Fiber composition percentage from scale bar
    meffColorX = [255,233,208,185] # Corresponding color intensities for these composition percentages
    
    ueffColors = [[219,252,246,249,255,237,239,255,254,254,255,254,254,254,255,252,255,252,251,250,254],[196,210,210,197,220,195,211,227,232,241,246,233,248,242,234,220,222,235,236,193,194]]
    
    ueffColorY = [0,10,20,30]  # Fiber composition percentage from scale bar
    ueffColorX = [255,244,230,213]# Corresponding color intensities for these composition percentages
    
    ###########################

    rng = np.random.default_rng(fascIdx) # Sets random seed
    
    if fascTypes[fascIdx]:
        side = 0
    else:
        side = 1
        
    maffFrac = distribution_params*.01 #sampleFractionHistogram(maffColorX,maffColorY,maffColors,side,rng)
    meffFrac = sampleFractionHistogram(meffColorX,meffColorY,meffColors,side,rng)
    ueffFrac = sampleFractionHistogram(ueffColorX,ueffColorY,ueffColors,side,rng)
    
    uaffFrac = 1-(maffFrac+meffFrac+ueffFrac)

    return maffFrac, meffFrac, ueffFrac, uaffFrac

def gammaDist(x,k,theta):
    
    return 1 / (gamma(k)*theta**k) * x**(k-1)*np.exp(-x/theta)

def prob(d, vals,smooth,distributionParams):
    
    binSizeSamples = np.diff(d)[0]
    
    empiricalDiams = vals[:,0]*1e-6 # From um to m
    empiricalProbs = vals[:,1]*0.01 # From percentage to fraction
    
    binSizeData = np.diff(empiricalDiams)[0] # Taking the first element ignores sloppy digitization towards the far end
    
    binRatio = binSizeSamples/binSizeData
    
    interp = interp1d(empiricalDiams,empiricalProbs,bounds_error=False,fill_value='extrapolate')
    
    interpD = interp(d)
    
    interpD[np.where(interpD<0)]=0
    
    if smooth:

        params = curve_fit(gammaDist,d*1e6,interpD*10,p0=[9,0.5])
                        
        #interpD = gammaDist(d*1e6,params[0][0],params[0][1]) * 0.1
        interpD = gammaDist(d*1e6,distributionParams[0],distributionParams[1]) * 0.1
    
#         N = 5
#         empiricalDiams = np.convolve(empiricalDiams, np.ones(N)/N, mode='valid')
#         empiricalProbs = np.convolve(empiricalProbs, np.ones(N)/N, mode='valid')
    
                      
    return interpD * binRatio


def MaffProb(d, maffProb):
    
    maffvals = np.loadtxt('../Data/maffvals.csv',delimiter=',')
    
    return maffProb * prob(d,maffvals,True,[5.7,0.59])

def MeffProb(d, meffProb):
    
    meffvals = np.loadtxt('../Data/meffvalsSmooth.csv',delimiter=',')
    
    return meffProb * prob(d,meffvals,True,[7.98,.428])

def UaffProb(d, uaffProb):
    
    uaffvals = np.loadtxt('../Data/uaffvals.csv',delimiter=',')
    
    return uaffProb * prob(d,uaffvals,False,distributionParams)

def UeffProb(d, ueffProb,distributionParams):
    
    ueffvals = np.loadtxt('../Data/ueffvals.csv',delimiter=',')
    
    return ueffProb * prob(d,ueffvals,True,distributionParams)


def sortTitrationSpace(table): 
    
    '''
    This function loads the titration results table from Sim4Life, and sorts it such that the splines and fascicles are in numerical order
    '''
    
    fascicles = []
    splines = []
    
    for column in table.columns:
                
        fasc = column.split('_')[0]
        
        try: # Get fascicle and spline name from column name
            fasc = int(fasc.split(' ')[-1])
            fascicles.append(fasc)
            spline = column.split(' [')[0].split('_')[-1]
            splines.append(int(spline))
        except: # Fasicle 0 is just called Fascicle_, not Fascicle_i like the other ones
            fascicles.append(0)
            spline = column.split(' [')[0].split('_')[-1]
            splines.append(int(spline))

    fascicles = np.array(fascicles)
    splines = np.array(splines)

    indices = np.lexsort((splines,fascicles))
    
    return table.iloc[:,indices]

def getFasciclePositions():
    
    '''
    Loads spline positions from sim4life. For each fascicle, average spline positions to get fascicle position
    '''
    
    fasciclePositions = []
    
    positions = np.load('../Data/fiberPositions1950.npy',allow_pickle=True)
    
    pos = positions[0][1]
    
    for i in np.arange(1,len(positions)):
        pos = np.vstack((pos,positions[i][1]))
    
    for i in range(39): # Selects positions for each fascicle and averages them
        
        fiberPos = pos[i*50:(i+1)*50]
        
        fasciclePositions.append(np.mean(fiberPos,axis=0))
        
    return np.array(fasciclePositions)
    
def getFascicleTypes(iteration=0):
    
    fascPos = getFasciclePositions()
    
    nerveCenter = np.mean(fascPos,axis=0)
    
    # Selects whether fascicle should have more afferent or more efferent fibers, based on whether it is left or right (or above vs below) dividing line
    
    if iteration == 0:
        fascTypes = fascPos[:,0] > 8
    elif iteration == 1:
        fascTypes = fascPos[:,0] < 8
    elif iteration == 2:
        fascTypes = fascPos[:,1] > -9
    else:
        fascTypes = fascPos[:,1] < -9
    
    return fascTypes

def getFibersPerFascicle(fascIdx,fascTypes,distribution_params):
    
    diamScaleFactor = getAreaScaleFactor()
        
    fascicleSizes = np.array([.24*.26,.16*.16,.18*.2,.16*.16,.12*.14,.16*.16,.1*.12,.24*.2,.2*.24,.18*.2,.14*.12,
                .16*.16,.1*.08,.16*.14,.12*.12,.08*.08,.14*.12,.1*.1,.2*.18,.14*.14,.14*.12,
                .12*.12,.22*.18,.14*.14,.14*.12,.18*.18,.16*.16,.1*.16,.12*.12,.22*.22,.1*.1,.1*.08,
                .12*.12,.1*.1,.12*.1,.14*.1,.1*.1,.14*.12,.18*.16])*1e-3**2
    
    maffFrac, meffFrac, uaffFrac, ueffFrac, fibersPerFascicle = getNumFibers(fascicleSizes,diamScaleFactor,fascIdx,fascTypes,distribution_params)
    
    return fibersPerFascicle # Average value

def Recruitment(current,diameters, fascIdx):
    
    d0Myelinated = 4e-6
    d0Unmyelinated = 0.8e-6
    
   
    #### Loads and sorts titration factors from S4L. Sorting is justg to make sure that fibers and fascicles are in numerical order (ie, fiber 0-fiber50, fascicle0-fascicle39)

    titrationFactorsMeff = sortTitrationSpace(pd.read_excel('../Data/TitrationGoodConductivity_Standoff_Sideways_HighConductivity.xlsx',index_col=0)).iloc[-1].values
    
    titrationFactorsUaff = sortTitrationSpace(pd.read_excel('../Data/TitrationGoodConductivity_Standoff_Sideways_Unmyelinated_HighConductivity.xlsx',index_col=0)).iloc[-1].values

    ####


    titrationFactors = [titrationFactorsMeff, titrationFactorsUaff]


    for j in [0,1]: # Myelinated and unmyelinated, respectively

        titrationFac = np.array(titrationFactors[j][fascIdx*50:(fascIdx+1)*50]) # Selects fibers in fascicle
        
        
        midptsX = np.sort(titrationFac)
        
        dupIdx =  np.where(np.diff(midptsX)==0)
        
        midptsX = np.delete(midptsX,dupIdx)
        
        cdfX = np.arange(0,len(midptsX))/len(midptsX)
        
        diff = np.diff(midptsX/midptsX[0])
                
        jumpIdx = np.where(diff > 1.25)[0]
                
        if fascIdx != 35 and j == 0 and len(jumpIdx)>0:
            if len(jumpIdx)>1:
                jumpIdx = jumpIdx[0]
                
            end = len(midptsX)
            jumpRange = np.arange(jumpIdx,end)

            midpts2 = np.delete(midptsX,jumpRange)
            cdf2 = np.arange(0,len(midpts2))/len(midpts2)
            
            cdfX = cdf2
            midptsX = midpts2
            
#         midptsX = np.insert(midptsX,0,0)
#         cdfX = np.insert(cdfX,0,0)
        
        if j == 0:
            
            midpts = midptsX
            cdf = cdfX

        if j == 1:
            
            midptsU = midptsX
            cdfU = cdfX

### Defines CDF of the titration curves
    
    interp = interp1d(midpts,cdf,bounds_error=False,fill_value=(0,1))

    interpU = interp1d(midptsU,cdfU,bounds_error=False,fill_value=(0,1))
##############
        
    myelinated = []
    unmyelinated = []


    for j, d in enumerate(diameters):

        myelinated.append(interp(current*(d/d0Myelinated)))

        unmyelinated.append(interpU(current*d/d0Unmyelinated))

    
    return [myelinated,unmyelinated]

def Scaling(d,fiberType): # Diameter dependent scaling of signal
    
    if fiberType == 0: # Myelinated fiber
        resistivity_intracellular = 0.7  # ohm meters
        deff = d # Calculates internal diameter from external diameter
        
    elif fiberType == 1: # Unmyelinated Fiber, Sundt Model
        resistivity_intracellular = 1
        deff = d
        
    elif fiberType == 2: # Unmyelinated fiber, tigerholm model
        resistivity_intracellular = 0.354 # ohm meters
        deff = d
    
    segment_length = 50e-6
    
    surfaceArea = segment_length * deff 
    
    xSectionArea = np.pi * (deff/2)**2
        
    resistance_intracellular = resistivity_intracellular/xSectionArea
        
    
    current_scale_factor = 1/(resistance_intracellular) 
                         
             
    return current_scale_factor


def editPhiShape(phi,distance):
    
    ''' 
    This function takes the recording exposure curve from S4L, shifts it to match the desired distance from stimulus to recording, and smooths it
    '''
    
    xvals = phi.iloc[:,0].values+distance -phi.iloc[np.argmax(phi.iloc[:,1].values),0] # Shift to match desired distance

    phiShapeEmpirical = phi.iloc[:,1].values-np.mean(phi.iloc[:,1])

    
   ######## 
    
    ####### Makes sure that the potential at the proximal end of the fiber goes all the way to zero

    if np.any(phiShapeEmpirical[:np.argmax(phiShapeEmpirical)]<0): # If the potential is negative at the end of the fiber, sets the potential to 0
        
        
        first = np.where(phiShapeEmpirical[:np.argmax(phiShapeEmpirical)]<0)[0][-1]    
        
        phiShapeEmpirical[:first] = 0
        
    else: # If the potential does not go all the way to 0 by the end of the fiber, forces it to zero
        
        first = np.where(np.abs(np.diff(phiShapeEmpirical))>1e-4)[0][0] # Based on derivative of function, selects point after whcih not to change values
        

        ### Linearizes potential before this point, up until it reaches 0 
        firsta = np.where(phiShapeEmpirical[first]-1e-4*np.arange(first)<0)[0][0]
        firsta = first-firsta

        phiShapeEmpirical[firsta:first] = 1e-4*np.arange(first-firsta)
        #######
        
        phiShapeEmpirical[0:firsta]=0 # Sets potential to zero
        
    ############
   
    #### Does the same kind of smoothing as above, but for the distal end fo the fiber
    if np.any(phiShapeEmpirical[np.argmin(phiShapeEmpirical):]>0):
                
        last = np.where(phiShapeEmpirical[np.argmin(phiShapeEmpirical):]>0)[0][0]+np.argmin(phiShapeEmpirical)
        
        phiShapeEmpirical[last:] = 0
        
    else:
        last = np.where(np.abs(np.diff(phiShapeEmpirical))>1e-4)[0][-1]
        lasta = np.where(phiShapeEmpirical[last]+1e-4*np.arange(len(phiShapeEmpirical)-last)>0)[0][0]
        lasta += last

        phiShapeEmpirical[last:lasta] = 1e-4* np.arange(lasta-last)+ phiShapeEmpirical[last]
        phiShapeEmpirical[lasta:] = 0
    

    return xvals, phiShapeEmpirical

def FitPhiShape(fascIdx,distance):
    
    ''' 
    This function creates an interpolation object for the recording exposure
    '''

    phi = pd.read_excel('../Data/PhiConductivity_Bipolar_Corrected/'+str(fascIdx)+'_BetterConductivity.xlsx')
    
    xvals, phiShapeEmpirical = editPhiShape(phi,distance)

    return interp1d(xvals,phiShapeEmpirical,bounds_error=False,fill_value=(phiShapeEmpirical[0],phiShapeEmpirical[-1]))

def PhiShape(velocity,t,function):
    
    
    '''
    This function stretches the recording exposure in time, based on fiber velocity
    '''

    phiOut = []
    
#     sos = butter(1, 20000, 'lp', fs=83333, output='sos') # We apply a low-pass filter to avoid numerical issues

    for i, v in enumerate(velocity): # Iterates through fibers

        x = t*v ### Defines interpolation points from time vector and fiber velocity

        out = function(x)
            
#         filtered = sosfilt(sos,out)


        phiOut.append(out)

    return np.array(phiOut)

def getVelocities(d0List,velocities,dList):

    velocityList = []

    for i, d in enumerate(d0List):
        
        if i < 1: # Myelinated velocities are linear with diameter
            d0 = d0List[0]
            v0 = velocities[0]
            
            velocity = v0 * dList/d0
            
        else: # Unmyelinated velocities go as the square root of diameter
            d0 = d0List[1]
            v0 = velocities[1]
            
            velocity = v0 * np.sqrt(dList/d0)

        
        velocityList.append(velocity)

    return velocityList


def PhiWeight(d, current,fascIdx, fascTypes, distribution_params):
    
    phiWeight = [ [[],[]] ]
    
    recruitment = Recruitment(current,d,fascIdx)
    
    scaling = []
    
    scalingFactors = [1,2]
    
    maffProb, meffProb, ueffProb, uaffProb = getFiberTypeFractions(fascIdx, fascTypes,distribution_params)
    
    numFibersPerFascicle = getFibersPerFascicle(fascIdx,fascTypes,distribution_params)
    
    
##### Weight is given by the product of the recruitment curve and the diameter probability curve
    phiWeight[0][0] =  MaffProb(d,maffProb)  * recruitment[0] * numFibersPerFascicle
    phiWeight[0][1] =  MeffProb(d,meffProb)  * recruitment[0] * numFibersPerFascicle
           
    
    return phiWeight,recruitment


def FitAPShape(ap,tphi): # Interpolates AP shape for a given AP
    
    
    # Ignores initial transient
    tv = ap.iloc[50:,0]
    v = ap.iloc[50:,1]
    
    ### Sets peak time to 0
    peak = tv[np.argmax(v)]
    tv -= peak


    apShapeEmpirical = v.values
        

    func = interp1d(tv,apShapeEmpirical,bounds_error=False,fill_value=(apShapeEmpirical[0],apShapeEmpirical[-1]))
    
    Vs = func(tphi)  
    
    
    #### Applies low-pass filter with very high cutoff, to remove artifacts
    sos = butter(1, 20000, 'lp', fs=83333, output='sos')
    
    V = sosfilt(sos,Vs)
    
    V[:10] = V[10]
         
    return V

def getDiameters():
    
   
    minDiam = .1
    
    
    maxDiam = 15 #7 + 5*iteration/30 
    
    d = np.linspace(minDiam,maxDiam,2000)*1e-6

    return d

def getPhiWeight(d, current,fascIdx,fascTypes,distribution_params):
    
    phiWeight = []
    recruitment = []
    
    for c in current:
        
        p, rec = PhiWeight(d,c,fascIdx,fascTypes,distribution_params)

        phiWeight.append(p)
        recruitment.append(rec)
        

    phiWeight0 = phiWeight[0][0][0][np.newaxis]
    phiWeight1 = phiWeight[0][0][1][np.newaxis]


    for i in np.arange(1,len(phiWeight)):
        phiWeight0 = np.vstack((phiWeight0,phiWeight[i][0][0][np.newaxis]))
        phiWeight1 = np.vstack((phiWeight1,phiWeight[i][0][1][np.newaxis]))

    
    return phiWeight0, phiWeight1

def getVs(aps,tphi): # Interpolates AP shapes in time for myelinated and unmyelinated fibers
    
    Vs = []

    for i, ap in enumerate(aps): # Iterates throguh the two classes (meylinated and unlyeminated)

        v = FitAPShape(ap,tphi)

        Vs.append(v)
        Vs.append(v)
        
    return Vs

    
def write_fascicle_signals(fascIdx,distribution_params):

    distribution_params = distribution_params[fascIdx]
     
    t= time.time()
   
    ## Selects fascicle and random seed for each available cpu


    #####
    
    
    distances = [0.06,.05,0.01] # Stimulus-recording distance, in m
    
    distanceIdx = 0 
    
    distance = distances[distanceIdx]
    
    ### Loads action potential shapes in time
    ap = pd.read_excel('../Data/APShape20.xlsx') # Rat
    ####

    aps = [ap]

    nx=50000

    tmin=-.5 # In s
    tmax=.5 # In s
    tphi=np.arange(tmin,tmax,(tmax-tmin)/(nx-1))

    current = np.array([500])/28.6

    recordingCurrent = 509e-6 # Current in the S4L recording simulation

    d0List = [20*1e-6] # Diameters of myelinated and unmyelinated fibers used to calculate velocities

    velocities = [86.95] # Velcities for the above diamters
    
    
    fascTypes = getFascicleTypes()# Defines whether fasicle is on left or right side of nerve

    d = getDiameters()


    phiWeight0, phiWeight1 = getPhiWeight(d,current,fascIdx, fascTypes,distribution_params) # Scaling factor for each diameter

    phi = [0,0]


    velocityList = getVelocities(d0List,velocities,d) # Gets velocity for each diameter


    names = ['myelinated','unmyelinated']


    phiFunc = FitPhiShape(fascIdx,distance)# Defines an interpolation function for the recording exposure for the fasicle
    ### For each diameter, defines a shifted and scaled exposure function
    phiShape0 = PhiShape(velocityList[0],tphi,phiFunc)
    
    
   #############


### Scales exposure functions

    scaling0 = (Scaling(d,0)* (tphi[1]-tphi[0])/velocityList[0])
        
    scaling00 = phiWeight0.T*scaling0[:,np.newaxis]
    scaling10 = phiWeight1.T*scaling0[:,np.newaxis]

    
    
    phi0 = np.matmul(phiShape0.T,scaling00)
    phi1 = np.matmul(phiShape0.T,scaling10)
    phi = np.array([phi0,phi1])
    
################

    signal = 0

    Vs = getVs(aps,tphi) # Interpolates action potential shape in time

    k = np.linspace(-1,1-(2.0/np.shape(Vs)[1]),np.shape(Vs)[1])

    signals = []

    for i, V in enumerate(Vs):  

        der = np.diff(V,n=2)/((tphi[1]-tphi[0])**2) # Second derivative of action potential shape

        cv = []
        
        for j in range(len(current)):
            
            c = fftconvolve(der,phi[i,:,j],mode='same') # Convolves second derivative with exposure
            
            cv.append(c)


        cv = np.array(cv)

        signals.append(cv)

    signals = np.array(signals)
    
    signals /= recordingCurrent

    return np.sum(signals,axis=0)


    #names = ['maff','meff']

    #for typeIdx in range(len(names)):

        #np.save('signals/'+names[typeIdx]+'/signals_'+str(fascIdx)+'.npy',signals[typeIdx])


if __name__=="__main__":

    iteration = int(sys.argv[1])
    
    x0s = np.linspace(5,7,num=6)
    y0s = np.linspace(0.5,0.7,num=6)

    x1s = np.linspace(7.5,9.5,num=6) 
    y1s = np.linspace(0.4,0.6,num=6)

    index = 0

    x0 = x0s[iteration]
    for y0 in y0s:
        for x1 in x1s:
            for y1 in y1s:

                fascicle_mapping = np.array([[x0,y0],[x1,y1]])# np.array([[5.7,.59],[7.98,.42]])
                write_fascicle_signals(fascicle_mapping,iteration,index)
                index += 1
