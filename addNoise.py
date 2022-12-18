import numpy as np


def add_Noise(data, distribution, mu_or_start, std_or_end, printData):
    """
    :param data: Inputdata
    :param distribution: Normal or uniform
    :param mu_or_start: mu or start of uniform
    :param std_or_end: std or end of uniform
    :param printData: Wether you want to print the data
    :return:
    """
    warning = 0
    for k in range(len(data)):

        for i in range(len(data[0])):
            if distribution == 'normal' or distribution == 'Normal' or distribution == 'gaussian' or distribution == 'Gaussian':
                noise = np.random.normal(mu_or_start, std_or_end, 1)
            elif distribution == 'uniform' or distribution == 'Uniform':
                noise = np.random.uniform(mu_or_start, std_or_end)
            else:
                print("Choose a proper distribution")
                print("You can choose normal or uniform")
                warning = 1
                break
            data[k][i] = data[k][i] + noise[0]
        if warning == 1:
            break
    if printData == 1:
        print(data)
    return data

