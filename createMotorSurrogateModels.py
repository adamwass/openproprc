import os
import pickle
import time
import numpy as np
import scipy as sc
import openmdao.api as om

def loadMotoCalcData(pathMotoCalcData):

    motoCalcDataMatlab = sc.io.loadmat(pathMotoCalcData)

    strVars = ['motors', 'motorDataHeaders', 'motorPerformanceDataHeaders']
    for var in strVars:
        motoCalcDataMatlab[var] = [string.strip(' ') for string in motoCalcDataMatlab[var]]

    motoCalcDataMatlab['motorData'] = motoCalcDataMatlab['motorData'].astype(float)
    motoCalcDataMatlab['motorPerformanceData'] = np.squeeze(motoCalcDataMatlab['motorPerformanceData']).tolist()
    motoCalcDataMatlab['motorPerformanceData'] = [data.astype(float) for data in motoCalcDataMatlab['motorPerformanceData']]

    motoCalcData = {}
    for idxMotor, motor in enumerate(motoCalcDataMatlab['motors']):
        motoCalcData[motor] = {}
        for idxMotorDataHeader, motorDataHeader in enumerate(motoCalcDataMatlab['motorDataHeaders']):
            motoCalcData[motor][motorDataHeader] = motoCalcDataMatlab['motorData'][idxMotor, idxMotorDataHeader]
        motoCalcData[motor]['motorPerformanceData'] = {}
        for idxMotorPerformanceDataHeader, motorPerformanceDataHeader in enumerate(motoCalcDataMatlab['motorPerformanceDataHeaders']):
            motoCalcData[motor]['motorPerformanceData'][motorPerformanceDataHeader] = motoCalcDataMatlab['motorPerformanceData'][idxMotor][:, idxMotorPerformanceDataHeader]
    
    return motoCalcData

def createMotorModel(motoCalcData, motor):

    motorPerformanceData = motoCalcData[motor]['motorPerformanceData']
    nPoints = motorPerformanceData['propDiameter'].size

    propDiameterBreakpoints = np.unique(motorPerformanceData['propDiameter'])
    throttleBreakpoints = np.unique(motorPerformanceData['throttle'])
    logicalKeep = np.zeros(nPoints, dtype = bool)
    
    for propDiameterBreakpoint in propDiameterBreakpoints:
        propPitchBreakpoints = np.unique(motorPerformanceData['propPitch'][motorPerformanceData['propDiameter'] == propDiameterBreakpoint])
        for propPitchBreakpoint in propPitchBreakpoints:
            for throttleBreakpoint in throttleBreakpoints:
                logicalBreakpoint = np.logical_and.reduce((motorPerformanceData['propDiameter'] == propDiameterBreakpoint, motorPerformanceData['propPitch'] == propPitchBreakpoint, motorPerformanceData['throttle'] == throttleBreakpoint))
                velocityBreakpoints = motorPerformanceData['velocity'][logicalBreakpoint]
                
                nBreakpoints = velocityBreakpoints.size
                nSamples = 10
                nIntervals = nSamples - 1

                if nBreakpoints > nSamples:
                    nRemove = nBreakpoints - nSamples
                    idxMiddle = nSamples // 2 - 1
                    idxOffMiddle = np.setdiff1d(np.arange(nIntervals), idxMiddle)
                    nOffMiddle = abs(idxOffMiddle - idxMiddle)
                    nPattern = 2*nOffMiddle - 1
                    nGroup = nRemove // nIntervals
                    idxGroupStart = nGroup*nIntervals

                    nSkip = np.zeros(nIntervals, dtype = int)
                    nSkip[idxMiddle] = nGroup + (nRemove - idxGroupStart) % 2
                    nSkip[idxOffMiddle] = np.ceil((nRemove - nPattern) / nIntervals)

                    nIncrement = nSkip + 1
                    idxKeep = np.cumsum(np.append(1, nIncrement)) - 1
                    logicalKeepLocal = np.zeros(nBreakpoints, dtype = bool)
                    logicalKeepLocal[idxKeep] = True

                else:
                    logicalKeepLocal = np.ones(nBreakpoints, dtype = bool)

                logicalKeep[logicalBreakpoint] = logicalKeepLocal

    motorPerformanceDataReduced = {k: v[logicalKeep] for (k, v) in motorPerformanceData.items()}

    motorModel = om.MetaModelUnStructuredComp()

    surrogateModelData = {
        'propDiameter': motorPerformanceDataReduced['propDiameter'],
        'propPitch': motorPerformanceDataReduced['propPitch'],
        'throttle': motorPerformanceDataReduced['throttle'],
        'velocity': motorPerformanceDataReduced['velocity'],
        'thrust': motorPerformanceDataReduced['thrust'],
        'inputPower': motorPerformanceDataReduced['inputPower']
    }

    motorModel.add_input('propDiameter', 0.0, training_data = motorPerformanceDataReduced['propDiameter'])
    motorModel.add_input('propPitch', 0.0, training_data = motorPerformanceDataReduced['propPitch'])
    motorModel.add_input('throttle', 0.0, training_data = motorPerformanceDataReduced['throttle'])
    motorModel.add_input('velocity', 0.0, training_data = motorPerformanceDataReduced['velocity'])

    dirSurrogateModels = os.path.join(os.path.dirname(__file__), 'surrogate_models')
    if not os.path.isdir(dirSurrogateModels):
        os.mkdir(dirSurrogateModels)
    dirTrainingCache = os.path.join(dirSurrogateModels, motor)
    if not os.path.isdir(dirTrainingCache):
        os.mkdir(dirTrainingCache)

    pathSurrogateModelData = os.path.join(dirTrainingCache, 'surrogate_model_data.pickle')
    pathThrustTrainingData = os.path.join(dirTrainingCache, 'thrust_training_data.dat')
    pathInputPowerTrainingData = os.path.join(dirTrainingCache, 'inputPower_training_data.dat')
    
    with open(pathSurrogateModelData, 'wb') as fileSurrogateModelData:
        pickle.dump(surrogateModelData, fileSurrogateModelData)

    motorModel.add_output('thrust', 0.0, training_data = motorPerformanceDataReduced['thrust'], surrogate = om.KrigingSurrogate(eval_rmse = True, lapack_driver = 'gesdd', training_cache = pathThrustTrainingData))
    motorModel.add_output('inputPower', 0.0, training_data = motorPerformanceDataReduced['inputPower'], surrogate = om.KrigingSurrogate(eval_rmse = True, lapack_driver = 'gesdd', training_cache = pathInputPowerTrainingData))

    motorModel.options['default_surrogate'] = om.KrigingSurrogate()

    prob = om.Problem()
    prob.model.add_subsystem('motorModel', motorModel)
    prob.setup()
    prob.run_model()

motoCalcData = loadMotoCalcData('/home/adamwass/Documents/mfly_mdo/motoCalcDataPython.mat')
motors = motoCalcData.keys()
nMotors = len(motors)
for idxMotor, motor in enumerate(motors):
    tStart = time.time()
    createMotorModel(motoCalcData, motor)
    tEnd = time.time()
    print(f'Motor ({idxMotor + 1:2d}/{nMotors:2d}) = {motor}')
    print(f'Training time = {tEnd - tStart:5.1f} s')
