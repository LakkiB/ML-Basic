package cs475.classify.generalizedlearningmodels;

import cs475.classify.Predictor;
import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.dataobject.label.ClassificationLabel;
import cs475.dataobject.label.Label;
import cs475.utils.CommandLineUtilities;

import java.util.HashMap;
import java.util.List;

public abstract class LinearThresholdClassifierBase extends Predictor
{

    protected LinearThresholdClassifierBase ( )
    {
        setWeightVectorW( new HashMap<Integer, Double>() );
        initializeParametersToDefaults();
        initializeParametersIfArgumentsProvided();
    }


    private void initializeParametersIfArgumentsProvided ( )
    {
        if(this instanceof WinnowPredictor)
            setLearningRateEeta(2.0);
        else if (this instanceof PerceptronPredictor)
            setLearningRateEeta(1.0);

        if (CommandLineUtilities.hasArg("online_learning_rate"))
            setLearningRateEeta(CommandLineUtilities.getOptionValueAsFloat("online_learning_rate"));


        setNoOfLearningIterationsI( 1 );
        if ( CommandLineUtilities.hasArg( "online_training_iterations" ) )
        {
            setNoOfLearningIterationsI( CommandLineUtilities.getOptionValueAsInt( "online_training_iterations" ) );
        }

        if ( CommandLineUtilities.hasArg( "thickness" ) )
        {
            thickness = CommandLineUtilities.getOptionValueAsFloat( "thickness" );
        }
    }

    public static int getTotalNoOfFeatures ( List<Instance> instances )
    {
        int maxIndex = 0;
        for ( Instance instance : instances )
        {
            for ( Integer featureIndex : instance.getFeatureVector().getFeatureVectorKeys() )
            {
                if ( featureIndex > maxIndex )
                {
                    maxIndex = featureIndex;
                }
            }
        }
        return maxIndex;
    }
    protected abstract void initializeWeights(List<Instance> instances);

    public abstract void initializeParametersToDefaults ( );

    protected abstract void updateWeight ( Label prediction, FeatureVector fv, HashMap<Integer,
            Double> weightVectorW, double learningRate );

    @Override
    public void train ( List<Instance> instances )
    {
        initializeWeights(instances);

        int learningIterations =  getNoOfLearningIterationsI();
        for ( int i = 0 ; i < learningIterations ; ++i )
        {
            for ( Instance instance : instances )
            {
                Label givenLabel = instance.getLabel();
                FeatureVector fv = instance.getFeatureVector();

                double linearCombinationWDotX = 0;

                for ( int feature: fv.getFeatureVectorKeys() )
                {
                    double weight = getWeightVectorW().get( feature );
                    linearCombinationWDotX += fv.get(feature) * weight;
                }

                if(givenLabel.getLabelValue() == 0)
                    givenLabel = new ClassificationLabel(-1);

                Label prediction = makePrediction( linearCombinationWDotX, scalarThresholdBeta, thickness );
                if ( prediction.getLabelValue() != givenLabel.getLabelValue() )
                {
                    updateWeight( givenLabel, fv, getWeightVectorW(), getLearningRateEeta() );
                }
            }
        }
    }


    protected Label makePrediction ( double wDotX, double scalarThresholdBeta, double thickness )
    {
        ClassificationLabel prediction;
        if ( wDotX >= scalarThresholdBeta + thickness )
        {
            prediction = new ClassificationLabel( 1 );
        }
        else if ( wDotX <= scalarThresholdBeta - thickness )
        {
            prediction = new ClassificationLabel( -1 );
        }
        else
        {
            prediction = new ClassificationLabel( 0 );
        }

        return prediction;
    }

    @Override
    public Label predict ( Instance instance )
    {
        double summationOfWDotX = 0;
        for ( int feature : instance.getFeatureVector().getFeatureVectorKeys() )
        {
            if(!getWeightVectorW().containsKey(feature))
                continue;

            double weight = getWeightVectorW().get( feature );
            summationOfWDotX += instance.getFeatureVector().get(feature) * weight;
        }

        Label prediction = makePrediction( summationOfWDotX, scalarThresholdBeta, thickness );

        if(prediction.getLabelValue() == -1)
            prediction = new ClassificationLabel(0);

        return prediction;
    }


    public double getLearningRateEeta ( )
    {
        return learningRateEeta;
    }

    public void setLearningRateEeta ( double learningRateEeta )
    {
        this.learningRateEeta = learningRateEeta;
    }

    public int getNoOfLearningIterationsI ( )
    {
        return noOfLearningIterationsI;
    }

    public void setNoOfLearningIterationsI ( int noOfLearningIterationsI )
    {
        this.noOfLearningIterationsI = noOfLearningIterationsI;
    }

    public HashMap<Integer, Double> getWeightVectorW ( )
    {
        return weightVectorW;
    }

    public void setWeightVectorW ( HashMap<Integer, Double> weightVectorW )
    {
        this.weightVectorW = weightVectorW;
    }

    protected double thickness;
    private double learningRateEeta;
    protected double scalarThresholdBeta;
    private int noOfLearningIterationsI;
    private HashMap<Integer, Double> weightVectorW;
}
