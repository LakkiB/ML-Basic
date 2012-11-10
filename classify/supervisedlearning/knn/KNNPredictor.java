package cs475.classify.supervisedlearning.knn;


import cs475.classify.Predictor;
import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.utils.CommandLineUtilities;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public abstract class KNNPredictor extends Predictor
{

    public KNNPredictor ( )
    {
        dataset      = new ArrayList<Instance>();
        seenFeatures = new HashSet<Integer>();
    }

    @Override
    public void train ( List<Instance> instances )
    {
        saveTrainingSetForPrediction( instances );
        saveSeenFeaturesForPrediction(instances);
    }

    private void saveTrainingSetForPrediction ( List<Instance> instances )
    {
        for ( Instance instance : instances )
        {
            try
            {
                dataset.add( ( Instance ) instance.clone() );
            }
            catch ( CloneNotSupportedException e )
            {
                e.printStackTrace();
            }
        }
    }

    private void saveSeenFeaturesForPrediction ( List<Instance> instances )
    {
        for ( Instance instance : instances )
        {
            for ( int feature : instance.getFeatureVector().getFeatureVectorKeys() )
            {
                seenFeatures.add( feature );
            }
        }
    }

    protected double computeDifferenceNorm ( FeatureVector input, FeatureVector trainingVectorX )
    {
        double distance = 0;
        for ( Integer feature : seenFeatures )
        {
            double inputFvVal = 0, trainingFvVal = 0;

            if ( input.getFeatureVectorKeys().contains( feature ) )
            {
                inputFvVal = input.get( feature );
            }
            if ( trainingVectorX.getFeatureVectorKeys().contains( feature ) )
            {
                trainingFvVal = trainingVectorX.get( feature );
            }
            distance += Math.pow( ( inputFvVal - trainingFvVal ), 2 );
        }
        return Math.sqrt( distance );
    }



    protected int getNumberOfNearestNeighbors ( )
    {
        int kNearestNeighbors = 5;
        if ( CommandLineUtilities.hasArg( "k_nn" ) )
        {
            kNearestNeighbors = CommandLineUtilities.getOptionValueAsInt( "k_nn" );
        }
        return kNearestNeighbors;
    }

    protected List<Instance> dataset;
    protected Set<Integer> seenFeatures;
}
