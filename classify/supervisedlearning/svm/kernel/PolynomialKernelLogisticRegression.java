package cs475.classify.supervisedlearning.svm.kernel;

import cs475.dataobject.FeatureVector;
import cs475.utils.CommandLineUtilities;

public class PolynomialKernelLogisticRegression extends KernelLogisticRegression {

    @Override
    protected double kernelFunction(FeatureVector fv1, FeatureVector fv2)
    {
        double polynomialKernelExponent = 2;
        if (CommandLineUtilities.hasArg("polynomial_kernel_exponent"))
            polynomialKernelExponent = CommandLineUtilities.getOptionValueAsFloat("polynomial_kernel_exponent");

        return Math.pow(1 + computeLinearCombination(fv1, fv2), polynomialKernelExponent);
    }
}
