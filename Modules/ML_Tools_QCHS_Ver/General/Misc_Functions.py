from sklearn.model_selection import train_test_split

def uncertRound(value, uncert):
    if uncert == 0:
        return value, uncert
    i = 0    
    while uncert*(10**i) <= 1:
        i += 1

    roundUncert = round(uncert, i)
    roundValue = round(value, i)
    if int(roundUncert) == roundUncert:
        roundUncert = int(roundUncert)
        roundValue = int(roundValue)
    return roundValue, roundUncert

def splitDevVal(inData, size=0.2, seed=1337):
    return train_test_split([i for i in inData.index.tolist()], test_size=size, random_state=seed)