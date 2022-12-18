import csv
import numpy as np
import math
import matplotlib.pyplot as plt


SSE_Matrix = [[]]
Tolerance = 0.2
Counter = 0
X = np.array(list(range(1, 251)))  # REMEMBER TO CHANGE NUMBER OF SAMPLES!
Fo = 0
Fi = 0
Co = 0
for j in range(20):
        for i in range(18):
                file = open(r'data\Data' + str(j+1)+'\Faulttype' + str(j+1)+'_initialcondition' + str(i+1) + '.csv')
                type(file)
                csvreader = csv.reader(file)

                rows = []
                for row in csvreader:
                        rows.append(row)
                file.close()

                data = np.array(rows)
                FloatData = data.astype(np.float64)

                DataMean = np.mean(FloatData, axis=0)

                temp = np.zeros(14)
                temp = temp.astype(np.float64)
                for k in range(14):
                        for h in range(len(FloatData)):
                                if np.divide(FloatData[h][k] - DataMean[k], DataMean[k], where=DataMean.any() != 0) > temp[k]:
                                        temp[k] = np.divide(FloatData[h][k] - DataMean[k], DataMean[k], where=DataMean.any() != 0)
                if j == 0 and i == 0:
                        SSE_Matrix[0] = temp
                else:
                        SSE_Matrix.append(temp)
SSE_Matrix = np.array(SSE_Matrix)
for i in range(len(SSE_Matrix)):
        for j in range(14):
                if abs(SSE_Matrix[i][j]) > Tolerance:
                        SSE_Matrix[i][j] = 1
                        aux = i / 18
                        Fo = math.trunc(aux)
                        Fi = i % 18
                        Co = j
                        Counter = Counter + 1
                        file = open(r'data\Data' + str(Fo+1)+'\Faulttype' + str(Fo+1)+'_initialcondition' + str(Fi+1) + '.csv')
                        csvreader1 = csv.reader(file)

                        rows1 = []
                        for row in csvreader1:
                                rows1.append(row)
                        file.close()

                        data1 = np.array(rows1)
                        FloatData1 = data1.astype(np.float64)
                        FloatData1 = FloatData1.T

                else:
                        SSE_Matrix[i][j] = 0
plt.show()
