import pandas
import numpy as np

def moveToCartesian(inData, particle, z=True, drop=False):
    try:
        pt = inData.loc[inData.index[:], particle + "_pT"]
        ptName = particle + "_pT"
    except KeyError:
        pt = inData.loc[inData.index[:], particle + "_pt"]
        ptName = particle + "_pt"

    if z: 
        eta = inData.loc[inData.index[:], particle + "_eta"]  

    phi = inData.loc[inData.index[:], particle + "_phi"]

    inData[particle + '_px'] = pt*np.cos(phi)
    inData[particle + '_py'] = pt*np.sin(phi)
    if z: 
        inData[particle + '_pz'] = pt*np.sinh(eta)

    if drop:
        inData.drop(columns=[ptName, particle + "_phi"], inplace=True)
        if z:
            inData.drop(columns=[particle + "_eta"], inplace=True)
        
def moveToPtEtaPhi(inData, particle):
    px = inData.loc[inData.index[:], particle + "_px"]
    py = inData.loc[inData.index[:], particle + "_py"]
    if 'mPT' not in particle: 
        pz = inData.loc[inData.index[:], particle + "_pz"]  

    inData[particle + '_pT'] = np.sqrt(np.square(px)+np.square(py))

    if 'mPT' not in particle: 
        inData[particle + '_eta'] = np.arcsinh(pz/inData.loc[inData.index[:], particle + '_pT'])

    inData[particle + '_phi'] = np.arcsin(py/inData.loc[inData.index[:], particle + '_pT'])
    inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] > 0), particle + '_phi'] = \
            np.pi - inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] > 0), particle + '_phi']
    inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] < 0), particle + '_phi'] = \
            -1 * (np.pi + inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] < 0), particle + '_phi'])         
    inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] == 0), particle + '_phi'] = \
            np.random.choice([-1*np.pi, np.pi], inData[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] == 0)].shape[0])
    
def deltaphi(a, b):
    return np.sign(b-a)*(np.pi - np.abs(np.abs(a-b) - np.pi))

def twist(dphi, deta):
    return np.arctan(np.abs(dphi/deta))

def addAbsMom(inData, particle, z=True):
    if z:
        inData[particle + '_|p|'] = np.sqrt(np.square(inData.loc[inData.index[:], particle + '_px']) +
                                            np.square(inData.loc[inData.index[:], particle + '_py']) +
                                            np.square(inData.loc[inData.index[:], particle + '_pz']))
    else:
        inData[particle + '_|p|'] = np.sqrt(np.square(inData.loc[inData.index[:], particle + '_px']) +
                                            np.square(inData.loc[inData.index[:], particle + '_py']))

def addEnergy(inData, particle):
    if particle + '_|p|' not in inData.columns:
        addAbsMom(inData, particle)

    inData[particle + '_E'] = np.sqrt(np.square(inData.loc[inData.index[:], particle + '_mass']) +
                                      np.square(inData.loc[inData.index[:], particle + '_|p|']))

def addMT(inData, pT, phi, name):
    inData[name + '_mT'] = np.sqrt(2 * pT * inData['mPT_pT'] * (1 - np.cos(deltaphi(phi, inData['mPT_phi']))))