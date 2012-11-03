package cs475.classify.generalizedlearningmodels;

import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;

import java.util.HashMap;
import java.util.List;

public class PerceptronPredictor extends LinearThresholdClassifierBase
{

    @Override
    public void initializeParametersToDefaults ( List<Instance> instances )
    {
        int n = getTotalNoOfFeatures( instances );
        for ( int i = 1 ; i <= n ; i++ )
        {
            getWeightVectorW().put( i, 0.0 );
        }

        thickness = 0;
        setLearningRateEeta( 1 );
        scalarThresholdBeta = 0;
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

    @Override
    protected void updateWeight (
            Label yi, FeatureVector fv, HashMap<Integer, Double> weightVectorW,
            double learningRate )
    {
        double yiValue = yi.getLabelValue();

        for ( Integer feature : fv.getFeatureVectorKeys() )
        {
            double oldWeight = weightVectorW.get( feature );
            double newWeight = oldWeight + learningRate * yiValue * fv.get( feature );
            weightVectorW.put( feature, newWeight );
        }
    }

}
