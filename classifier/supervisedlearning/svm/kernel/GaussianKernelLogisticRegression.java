package cs475.classifier.supervisedlearning.svm.kernel;

import cs475.dataobject.FeatureVector;
import cs475.utils.CommandLineUtilities;
import cs475.utils.UtilityFunctions;


public class GaussianKernelLogisticRegression extends KernelLogisticRegression
{

    public GaussianKernelLogisticRegression ()
    {
        gaussianKernelSigma = 1;
        if ( CommandLineUtilities.hasArg( "gaussian_kernel_sigma" ) )
        {
            gaussianKernelSigma = CommandLineUtilities.getOptionValueAsFloat( "gaussian_kernel_sigma" );
        }
    }

    protected double kernelFunction ( FeatureVector fv1, FeatureVector fv2 )
    {
        return Math.exp( -1 * UtilityFunctions.computeL2Norm( fv1, fv2 ) / 2 * gaussianKernelSigma );
    }

    private double gaussianKernelSigma;
}
