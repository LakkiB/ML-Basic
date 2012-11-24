package cs475.classifier.supervisedlearning.svm.kernel;

import cs475.dataobject.FeatureVector;
import cs475.utils.CommandLineUtilities;
import cs475.utils.UtilityFunctions;


public class GaussianKernelLogisticRegression extends KernelLogisticRegression {
    @Override

    protected double kernelFunction(FeatureVector fv1, FeatureVector fv2)
    {
        double gaussianKernelSigma = 1;
        if (CommandLineUtilities.hasArg("gaussian_kernel_sigma"))
            gaussianKernelSigma = CommandLineUtilities.getOptionValueAsFloat("gaussian_kernel_sigma");
        if(gaussianKernelSigma == 0)
            gaussianKernelSigma = 1;

        return Math.exp(-1 * UtilityFunctions.computeL2Norm( fv1, fv2 )/2 * gaussianKernelSigma);
    }

   /* private double computeSquareNorm(FeatureVector fv1, FeatureVector fv2)
    {
        double squareNorm = 0;

        for(Integer feature : fv1.getFeatureVectorKeys())
        {
            if(fv2.getFeatureVectorKeys().contains(feature))
            {
                squareNorm += Math.pow(fv1.get(feature) - fv2.get(feature), 2);
            }
            else
            {
                squareNorm += Math.pow(fv1.get(feature), 2);
            }
        }

        for(Integer feature : fv2.getFeatureVectorKeys())
        {
            if(!fv1.getFeatureVectorKeys().contains(feature))
                squareNorm += Math.pow(0 - fv2.get(feature), 2);
        }
        return Math.sqrt(squareNorm);
    }*/

}
