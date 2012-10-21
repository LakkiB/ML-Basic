package cs475.classify.svm.kernel;

import cs475.dataobject.FeatureVector;

public class LinearKernelLogisticRegression extends KernelLogisticRegression{


    protected double kernelFunction(FeatureVector fv1, FeatureVector fv2) {
        return computeLinearCombination
                (fv1, fv2);
    }

}
