package cs475.classifier.supervisedlearning.knn;


import cs475.classifier.Predictor;
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

    private void setKnn()
    {
        System.out.println("In set k_nn");
        kNearestNeighbors = DEFAULT_KNN;
        if ( CommandLineUtilities.hasArg( "k_nn" ) )
        {
            kNearestNeighbors = CommandLineUtilities.getOptionValueAsInt( "k_nn" );
            System.out.println("Setting k_nn to " + kNearestNeighbors);
        }
    }

    @Override
    public void train ( List<Instance> instances )
    {
        setKnn();
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


    protected int kNearestNeighbors;
    protected List<Instance> dataset;
    protected Set<Integer> seenFeatures;
    public static final int DEFAULT_KNN = 5;
}
