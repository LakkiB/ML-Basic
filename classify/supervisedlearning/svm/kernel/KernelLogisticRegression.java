package cs475.classify.supervisedlearning.svm.kernel;

import cs475.classify.Predictor;
import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.dataobject.label.ClassificationLabel;
import cs475.dataobject.label.Label;
import cs475.utils.CommandLineUtilities;

import java.util.List;

public abstract class KernelLogisticRegression extends Predictor{

    public KernelLogisticRegression()
    {
        iterations  = 5;
        if (CommandLineUtilities.hasArg("gradient_ascent_training_iterations"))
            iterations =
                    CommandLineUtilities.getOptionValueAsInt("gradient_ascent_training_iterations");

        learningRate = 0.01;
        if (CommandLineUtilities.hasArg("gradient_ascent_learning_rate"))
            learningRate =
                    CommandLineUtilities.getOptionValueAsFloat("gradient_ascent_learning_rate");
    }

    @Override
    public void train(List<Instance> instances)
    {
        trainingInstances = instances;

        initializeGramMatrixAndAlphaVector(trainingInstances.size());
        computeGramMatrix(trainingInstances, gramMatrix, trainingInstances.size());

        while (iterations-- > 0)
        {
            preCacheAlphaDotGramMatrix(alphaVector, gramMatrix);
            for (int k = 0; k < alphaVector.length; k++)
            {
                double gradient = computeGradient(trainingInstances, k);
                updateAlphaVector(k, alphaVector, learningRate, gradient);
            }
        }
    }

    private void preCacheAlphaDotGramMatrix(double[] alphaVector, double[][] gramMatrix) {
        for (int i = 0; i < gramMatrix.length; i++)
        {
            double summation = 0;
            for (int j = 0; j < gramMatrix.length; j++)
            {
                summation += alphaVector[j] * gramMatrix[j][i];
            }
            preCachedAlphaDotGramMatrix[i] = summation;
        }
    }

    @Override
    public Label predict(Instance instance) {
        double linkFunctionOutput = 0;
        double summation = 0;
        for(int j = 0 ; j < trainingInstances.size() ; ++j)
        {
            summation += alphaVector[j] * kernelFunction(trainingInstances.get(j).getFeatureVector(), instance.getFeatureVector());
            linkFunctionOutput = linkFunction(summation);
        }
        return linkFunctionOutput >= 0.5? new ClassificationLabel(1): new ClassificationLabel(0);
    }

    protected abstract double kernelFunction(FeatureVector fv1, FeatureVector fv2);


    protected void computeGramMatrix(List<Instance> instances, double[][] gramMatrix, int noOfInstances) {
        for(int i = 0; i < noOfInstances ; ++i)
        {
            for (int j = 0 ; j < noOfInstances; ++j)
            {
                gramMatrix[i][j] = kernelFunction(instances.get(i).getFeatureVector(), instances.get(j).getFeatureVector());
            }
        }
    }


    private void initializeGramMatrixAndAlphaVector(int noOfInstances) {
        alphaVector                 = new double[noOfInstances];
        gramMatrix                  = new double[noOfInstances][noOfInstances];
        preCachedAlphaDotGramMatrix = new double[noOfInstances];
    }


    protected double computeGradient(List<Instance> instances, int k) {
        double gradient = 0;
        for (int i = 0; i < instances.size(); i++)
        {
            double labelValue = instances.get(i).getLabel().getLabelValue();
            if (labelValue == 1)
            {
                double linkFunctionOutput = linkFunction(-1 * preCachedAlphaDotGramMatrix[i]);
                gradient +=  linkFunctionOutput * gramMatrix[i][k];
            }
            else if (labelValue == 0)
            {
                double linkFunctionOutput = linkFunction(preCachedAlphaDotGramMatrix[i]);
                gradient += linkFunctionOutput * (-1 * gramMatrix[i][k]);
            }
        }
        return gradient;
    }

    private double linkFunction(double input) {
        return  1.0/(1.0 + Math.exp(-1.0 * input));
    }

    protected void updateAlphaVector(int k, double[] alphaVector, double learningRate, double gradient)
    {
        alphaVector[k] += learningRate * gradient;
    }

    protected double computeLinearCombination(FeatureVector fv1, FeatureVector fv2) {
        double dotProduct = 0.0;

        for (Integer feature : fv1.getFeatureVectorKeys())
            if(fv2.getFeatureVectorKeys().contains(feature))
                dotProduct += fv1.get(feature) * fv2.get(feature);

        return dotProduct;
    }

    protected int iterations;
    protected double learningRate;
    protected double[] alphaVector;
    protected double[][] gramMatrix;
    protected List<Instance> trainingInstances;
    protected  double [] preCachedAlphaDotGramMatrix;

}
