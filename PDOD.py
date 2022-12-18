import loadfile as lf
import numpy as np
import os
aux = np.array([96, 4, 80, 20, 60, 40])
LenghtOfData = 250
P = (aux/100)*LenghtOfData
faults = np.array(range(0, 21))
for j in range(21):
    Data = lf.load_file(j, 0, 'true', 'array')
    for i in range(int(len(P)/2)):
        if j == 9:
            bot = 15
        else:
            bot = 18
        for k in range(bot):
            kata = Data[k * 250:k * 250 + 250, :]
            boolLabel = 0
            if j != 0:
                boolLabel = 1
            labelBinary = np.full(shape=250, fill_value=boolLabel)
            labelSpecific = np.full(shape=250, fill_value=j)
            idx = np.vstack((labelBinary, labelSpecific))
            matrix = np.concatenate((kata, idx.T),axis=1)
            np.random.shuffle(matrix)

            if i == 0: # 96, 4
                trainData96 = matrix[0:int(P[i]), :]
                validationData2 = matrix[int(P[i]): LenghtOfData, :]
                if k == 0:
                    trainData96full = trainData96
                    validationData2full = validationData2
                else:
                    trainData96full = np.concatenate((trainData96full, trainData96), axis=0)
                    validationData2full = np.concatenate((validationData2full, validationData2), axis=0)

            elif i == 1: # 80, 20
                trainData80 = matrix[0:int(P[i*2]), :]
                validationData10 = matrix[int(P[i*2]): LenghtOfData, :]
                if k == 0:
                    trainData80full = trainData80
                    validationData10full = validationData10
                else:
                    trainData80full = np.concatenate((trainData80full, trainData80), axis=0)
                    validationData10full = np.concatenate((validationData10full, validationData10), axis=0)

            elif i == 2: # 60, 40
                trainData60 = matrix[0:int(P[i*2]), :]
                validationData20 = matrix[int(P[i*2]): LenghtOfData, :]
                if k == 0:
                    trainData60full = trainData60
                    validationData20full = validationData20
                else:
                    trainData60full = np.concatenate((trainData60full, trainData60), axis=0)
                    validationData20full = np.concatenate((validationData20full, validationData20), axis=0)

        temp = str(aux[i*2])
        basePath = r'PercentageWiseData\Faulttype' + str(j) + '\Percentage_' + temp
        os.makedirs(basePath, exist_ok=True)
        path = os.path.join(basePath, 'Train_' + temp + '.csv')

        if i == 0:
            path = os.path.join(basePath, 'Train_' + temp + '.csv')
            np.savetxt(path, trainData96full, delimiter=',', fmt='%s')
            path = os.path.join(basePath, 'Val_' + temp + '.csv')
            np.savetxt(path, validationData2full, delimiter=',', fmt='%s')

        elif i == 1:
            path = os.path.join(basePath, 'Train_' + temp + '.csv')
            np.savetxt(path, trainData80full, delimiter=',', fmt='%s')
            path = os.path.join(basePath, 'Val_' + temp + '.csv')
            np.savetxt(path, validationData10full, delimiter=',', fmt='%s')

        elif i == 2:
            path = os.path.join(basePath, 'Train_' + temp + '.csv')
            np.savetxt(path, trainData60full, delimiter=',', fmt='%s')
            path = os.path.join(basePath, 'Val_' + temp + '.csv')
            np.savetxt(path, validationData20full, delimiter=',', fmt='%s')

