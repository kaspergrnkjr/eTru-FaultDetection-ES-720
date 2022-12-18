This folder contains all the results from the validation of the Ml algorithms.

The naming convention are as follows:

Onstage  --> Full fault classification
Reduced  --> Full fault classification without faults 8 and 18
TwoStage --> One binary stage followed by a full fault classification without the non-faulty
RM 	 --> If 1 three features are removed otherwise kept
Binary	 --> If 1 then only binary classification is performed
SwapData --> If 1 then the data becomes erroneous 
Fraction --> This parameter tells how the data was balanced (non-faulty/faulty)
Gamma    --> Gamma parameter for the gaussian kernel
C	 --> Regularization parameter for the SVM classifier