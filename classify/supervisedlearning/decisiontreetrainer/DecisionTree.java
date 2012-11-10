package cs475.classify.supervisedlearning.decisiontreetrainer;

import cs475.classify.Predictor;
import cs475.classify.supervisedlearning.simpleclassifier.MajorityClassifier;
import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;
import cs475.utils.CommandLineUtilities;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class DecisionTree extends Predictor
{

    private Node rootNode;
    private List<Instance> trainingInstances;
    private HashMap<Integer, Boolean> featureTypes;
    private int totalNoOfFeaturesInTrainingSet;
    private HashMap<Integer, Double> precomputedMeans;

    public DecisionTree ( )
    {
        rootNode = null;
        trainingInstances = new ArrayList<Instance>();
        featureTypes = new HashMap<Integer, Boolean>();
        precomputedMeans = new HashMap<Integer, Double>();
    }


    private void preComputeAndCacheMeans ( int totalFeatures, List<Instance> instances )
    {
        for ( int i = 1 ; i <= totalFeatures ; ++i )
        {
            precomputedMeans.put( i, 0.0 );
        }

        for ( Instance instance : instances )
        {
            for ( Integer feature : instance.getFeatureVector().getFeatureVectorKeys() )
            {
                precomputedMeans.put( feature, precomputedMeans.get( feature ) + instance.getFeatureVector().get(
                        feature ) );
            }
        }
        for ( int index : precomputedMeans.keySet() )
        {
            precomputedMeans.put( index, precomputedMeans.get( index ) / instances.size() );
        }
    }

    public Node buildDecisionTree ( List<Instance> instancesInCurrentTree,
                                    int maxDepth ) throws CloneNotSupportedException
    {
        // no instances to work on?
        if ( instancesInCurrentTree.size() == 0 )
        {
            Label majority = getMajorityLabel( trainingInstances );
            return new Node( majority, true );
        }
        // can't split ?
        if ( maxDepth == 0 )
        {
            Label majority = getMajorityLabel( instancesInCurrentTree );
            return new Node( majority, true );
        }
        // All labels are equal?
        boolean allLabelsEqual = checkIfAllLabelsAreEqual( instancesInCurrentTree );
        if ( allLabelsEqual )
        {
            Label predictLabel = instancesInCurrentTree.get( 0 ).getLabel();
            return new Node( predictLabel, true );
        }

        List<Instance> leftSubTree = new ArrayList<Instance>();
        List<Instance> rightSubtree = new ArrayList<Instance>();

        preComputeAndCacheMeans( featureTypes.size(), instancesInCurrentTree );
        int featureIndex = getUniqueFeatureToSplitOn( instancesInCurrentTree, precomputedMeans );
        if ( isNoMoreFeaturesToSplit( featureIndex ) )
        {
            Label majorityLabel = getMajorityLabel( instancesInCurrentTree );
            return new Node( majorityLabel, true );
        }

        double mean = precomputedMeans.get( featureIndex );
        divideFeatureVectorsBasedOnMean( instancesInCurrentTree, leftSubTree, rightSubtree, mean, featureIndex );
        if ( leftSubTree.size() == instancesInCurrentTree.size() || rightSubtree.size() == instancesInCurrentTree
                .size() )
        {
            return new Node( getMajorityLabel( instancesInCurrentTree ), true );
        }


        Node newNode = new Node( mean, featureIndex );
        newNode.left = buildDecisionTree( leftSubTree, maxDepth - 1 );
        newNode.right = buildDecisionTree( rightSubtree, maxDepth - 1 );

        return newNode;
    }

    private boolean isNoMoreFeaturesToSplit ( int featureIndex )
    {
        return featureIndex == -1;
    }

    private int getUniqueFeatureToSplitOn ( List<Instance> instances, HashMap<Integer, Double> precomputedMeans )
    {
        return new C45DecisionTreeTrainer().getFeatureWithLeastEntropy( instances, featureTypes, precomputedMeans );
    }

    private Label getMajorityLabel ( List<Instance> instances )
    {
        Predictor majorityClassifier = new MajorityClassifier();
        majorityClassifier.train( instances );
        return majorityClassifier.predict( null );
    }

    private void divideFeatureVectorsBasedOnMean
            (
                    List<Instance> instances, List<Instance> leftSubTree,
                    List<Instance> rightSubtree,
                    double mean,
                    int featureIndex
            ) throws CloneNotSupportedException
    {

        for ( Instance instance : instances )
        {
            double featureVal = 0;
            Instance instanceClone = ( Instance ) instance.clone();

            if ( instance.getFeatureVector().getFeatureVectorKeys().contains( featureIndex ) )
            {
                featureVal = instance.getFeatureVector().get( featureIndex );
            }

            instanceClone.getFeatureVector().featureVector.remove( featureIndex );
            if ( featureVal <= mean )
            {
                leftSubTree.add( instanceClone );
            }
            else
            {
                rightSubtree.add( instanceClone );
            }
        }
    }

    private boolean checkIfAllLabelsAreEqual ( List<Instance> instances )
    {
        Label lastSeenLabel = null;
        boolean allLabelsEqual = true;
        for ( Instance instance : instances )
        {
            if ( lastSeenLabel == null )
            {
                lastSeenLabel = instance.getLabel();
            }
            else if ( instance.getLabel().getLabelValue() != lastSeenLabel.getLabelValue() )
            {
                allLabelsEqual = false;
                break;
            }
        }
        return allLabelsEqual;
    }

    @Override
    public void train ( List<Instance> instances )
    {
        try
        {
            for ( Instance instance : instances )
            {
                trainingInstances.add( ( Instance ) instance.clone() );
            }

            makeBinaryOrContinuousFeatureClassification( instances );
            rootNode = this.buildDecisionTree( instances, getMaxDepthArgument() );
            cleanUp();
        }
        catch ( CloneNotSupportedException e )
        {
            e.printStackTrace();
        }
    }

    private void makeBinaryOrContinuousFeatureClassification ( List<Instance> instances )
    {
        totalNoOfFeaturesInTrainingSet = getTotalNoOfFeatures( instances );
        markAllFeaturesAsBinary();

        for ( Instance instance : instances )
        {
            for ( Integer feature : instance.getFeatureVector().getFeatureVectorKeys() )
            {
                if ( !isBinary( feature ) ) continue; // Already set, nothing to do

                double featureValue = instance.getFeatureVector().get( feature );
                if ( featureValue != 0.0 && featureValue != 1.0 )
                {
                    setBinary( feature );
                }
            }
        }
    }

    private void setBinary ( Integer feature )
    {
        featureTypes.put( feature, false );
    }

    private boolean isBinary ( Integer feature )
    {
        return featureTypes.get( feature );
    }

    private void markAllFeaturesAsBinary ( )
    {
        for ( int i = 1 ; i <= totalNoOfFeaturesInTrainingSet ; i++ )
        {
            featureTypes.put( i, true );
        }
    }


    private void cleanUp ( )
    {
        trainingInstances.clear();
        precomputedMeans.clear();
        featureTypes.clear();
    }

    private int getTotalNoOfFeatures ( List<Instance> instances )
    {
        int maxIndex = 1;
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

    private int getMaxDepthArgument ( )
    {
        int max_decision_tree_depth = 4;
        if ( CommandLineUtilities.hasArg( "max_decision_tree_depth" ) )
        {
            max_decision_tree_depth = CommandLineUtilities.getOptionValueAsInt( "max_decision_tree_depth" );
        }
        return max_decision_tree_depth;
    }

    /* @Override
    public Label predict(Instance instance)
    {
        Node treeNode = rootNode;
        while (!treeNode.isLeaf )
        {
            double instanceFeatureValue = 0;
            if(instance.getFeatureVector().featureVector.containsKey(treeNode.featureToSplitOn))
                instanceFeatureValue = instance.getFeatureVector().get(treeNode.featureToSplitOn);

            if (instanceFeatureValue <= treeNode.mean)
                treeNode = treeNode.left;
            else
                treeNode = treeNode.right;
        }
        return  treeNode.prediction;
    }*/


    public Label predict ( Instance instance )
    {
        Node node = this.rootNode;
        while ( !node.isLeaf )
        {
            double featureValue = 0.0;
            try
            {
                featureValue = instance.getFeatureVector().get( node.featureToSplitOn );

            }
            catch ( Exception ignored )
            {
            }
            if ( featureValue == 0 || featureValue <= node.mean )
            {
                node = node.left;
            }
            else
            {
                node = node.right;
            }
        }
        return node.prediction;
    }


    class Node implements Serializable
    {
        private double mean;
        private int featureToSplitOn;
        private boolean isLeaf;
        private Label prediction;
        Node left;
        Node right;


        public Node ( Double mean, int featureIndex )
        {
            this.mean = mean;
            this.featureToSplitOn = featureIndex;
            this.isLeaf = false;
            this.prediction = null;
        }

        public Node ( Label majorityLabel, boolean isLeaf )
        {
            this.isLeaf = isLeaf;
            this.prediction = majorityLabel;

            // why am I doing this?
            this.mean = 0;
            this.featureToSplitOn = -1;
        }
    }

}
