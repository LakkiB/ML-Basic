package cs475.classify;

import cs475.dataobject.Instance;
import cs475.dataobject.label.ClassificationLabel;
import cs475.dataobject.label.Label;
import cs475.utils.CommandLineUtilities;

import java.io.Serializable;
import java.util.*;

public class DecisionTree extends Predictor{

    private Node rootNode;
    private Label majorityLabel;
    private Set<Integer> splits;
    private List<Instance> trainingInstances;

    public DecisionTree() {
        rootNode = null;
        majorityLabel = null;
        trainingInstances = null;
        splits = new HashSet<Integer>();
    }

    public Node buildDecisionTree(List<Instance> instances, int maxDepth) {

        // no instances to work on?
        if (instances.size() == 0) {
            majorityLabel = populateMajorityLabel(trainingInstances);
            return  new Node(majorityLabel, true);
        }

        // All labels are equal?
        boolean allLabelsEqual = checkIfAllLabelsAreEqual(instances);
        if (allLabelsEqual) {
            Label predictLabel = predictLabelWhenAllLabelsAreEqual(instances, allLabelsEqual);
            return new Node(predictLabel, true);
        }

        List<Instance> leftSubTree  = new ArrayList<Instance>();
        List<Instance> rightSubtree = new ArrayList<Instance>();
        int featureIndex            = getUniqueFeatureToSplitOn(instances);
        double meanOfThisFeature    = computeMeanForFeature(featureIndex, instances);

        divideFeatureVectorsBasedOnMean(instances, leftSubTree, rightSubtree, meanOfThisFeature, featureIndex);

        // can't split ?
        if (instances.size() == leftSubTree.size() || instances.size() == rightSubtree.size() || maxDepth == 0) {
            Label majority = populateMajorityLabel(instances);
            return new Node(majority, true);
        }
        // Build decision tree
        Node newNode = new Node(meanOfThisFeature, featureIndex);
        newNode.left = buildDecisionTree(leftSubTree, maxDepth - 1);
        newNode.right = buildDecisionTree(rightSubtree, maxDepth - 1);

        return newNode;
    }

    private int getUniqueFeatureToSplitOn(List<Instance> instances) {
        int featureIndex;
        do
        {
            featureIndex = new C45DecisionTreeTrainer().getFeatureWithLeastEntropy(instances);
        } while (splits.contains(featureIndex));

        splits.add(featureIndex);
        return featureIndex;
    }

    private Label populateMajorityLabel(List<Instance> instances) {
        Predictor majorityClassifier = new MajorityClassifier();
        majorityClassifier.train(instances);
        return majorityClassifier.predict(null);
    }

    private Label predictLabelWhenAllLabelsAreEqual(List<Instance> instances, boolean allLabelsEqual) {
        Label predictLabel = new ClassificationLabel(-1);
        if(allLabelsEqual && instances.listIterator().hasNext())
            predictLabel = instances.listIterator().next().getLabel();
        return predictLabel;
    }

    private void divideFeatureVectorsBasedOnMean
            (List<Instance> instances, List<Instance> leftSubTree, List<Instance> rightSubtree, double mean, int featureIndex) {
        for (Instance instance : instances) {
            if (instance.getFeatureVector().get(featureIndex) <= mean)
                leftSubTree.add(instance);
            else
                rightSubtree.add(instance);
        }
    }

    private Double computeMeanForFeature(Integer featureIndex, List<Instance> instances) {
        Double sum = 0.0;
        for (Instance instance:instances)
            sum += instance.getFeatureVector().get(featureIndex);
        return  sum/instances.size();
    }

    private boolean checkIfAllLabelsAreEqual(List<Instance> instances) {
        Label lastSeenLabel = null;
        boolean allLabelsEqual = true;
        for (Instance instance : instances) {
            if (lastSeenLabel == null)
                lastSeenLabel = instance.getLabel();
            else if (instance.getLabel().getLabelValue() != lastSeenLabel.getLabelValue()) {
                allLabelsEqual = false;
                break;
            }
        }
        return allLabelsEqual;
    }

    @Override
    public void train(List<Instance> instances) {
        trainingInstances = instances;
        int max_decision_tree_depth = 8;
        if (CommandLineUtilities.hasArg("max_decision_tree_depth"))
            max_decision_tree_depth = CommandLineUtilities.getOptionValueAsInt("max_decision_tree_depth");
        rootNode = this.buildDecisionTree(instances, max_decision_tree_depth);
    }

    @Override
    public Label predict(Instance instance) {
        Node treeNode = rootNode;

        while (!treeNode.isLeaf && (treeNode.left != null || treeNode.right != null) ) {
            double instanceFeatureValue = instance.getFeatureVector().get(treeNode.featureIndex);
            if (instanceFeatureValue <= treeNode.mean)
                treeNode = treeNode.left;
            else
                treeNode = treeNode.right;
        }
        return  treeNode.prediction;
    }


    class Node implements Serializable {
        private double mean;
        private int featureIndex;
        private boolean isLeaf;
        private Label prediction;
        Node left;
        Node right;


        public Node(Double mean, int featureIndex) {
            this.mean = mean;
            this.featureIndex = featureIndex;
            this.isLeaf = false;
            this.prediction = null;
        }

        public Node(Label majorityLabel, boolean isLeaf) {
            this.isLeaf = isLeaf;
            this.prediction = majorityLabel;

            // why am I doing this?
            this.mean = 99999999.0;
            this.featureIndex = -999999999;
        }
    }
}
