package cs475.classify.decisiontreetrainer;

import cs475.classify.simpleclassifier.MajorityClassifier;
import cs475.classify.Predictor;
import cs475.dataobject.Instance;
import cs475.dataobject.label.ClassificationLabel;
import cs475.dataobject.label.Label;
import cs475.utils.CommandLineUtilities;

import java.io.Serializable;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

public class DecisionTree extends Predictor {

    private Node rootNode;
    private Label majorityLabel;
    private List<Instance> trainingInstances;
    private List<Integer> usedFeatures = new ArrayList<Integer>();

    public DecisionTree() {
        rootNode = null;
        majorityLabel = null;
        trainingInstances = new ArrayList<Instance>();
    }

    public Node buildDecisionTree(List<Instance> instances, int maxDepth) {

        // no instances to work on?
        if (instances.size() == 0) {
            majorityLabel = populateMajorityLabel(trainingInstances);
            System.out.println(MessageFormat.format("No instances to work on. Returning leaf with majority label {0}", majorityLabel));
            return  new Node(majorityLabel, true);
        }

        List<Instance> leftSubTree  = new ArrayList<Instance>();
        List<Instance> rightSubtree = new ArrayList<Instance>();
        int featureIndex            = getUniqueFeatureToSplitOn(instances, usedFeatures);
        double meanOfThisFeature    = computeMeanForFeature(featureIndex, instances);

        divideFeatureVectorsBasedOnMean(instances, leftSubTree, rightSubtree, meanOfThisFeature, featureIndex);

        // can't split ?
        if (instances.size() == leftSubTree.size() || instances.size() == rightSubtree.size() || maxDepth == 0) {
            Label majority = populateMajorityLabel(instances);
            System.out.println(MessageFormat.format(MessageFormat.format
                    ("Can''t split anymore. skewed? maxDepth is {0}. Returning leaf with majority label {0}", maxDepth), majority));
            return new Node(majority, true);
        }

        System.out.println(MessageFormat.format("left subtree size = {0}, right subtree size = {1}", leftSubTree.size(), rightSubtree.size()));

        // All labels are equal?
        boolean allLabelsEqual = checkIfAllLabelsAreEqual(instances);
        if (allLabelsEqual) {
            Label predictLabel = predictLabelWhenAllLabelsAreEqual(instances, allLabelsEqual);
            System.out.println(MessageFormat.format("All labels are equal. Returning leaf with majority label {0}", predictLabel));
            return new Node(predictLabel, true);
        }

        removeFeatureFromFeatureVector(instances, featureIndex);

        // Build decision tree
        Node newNode = new Node(meanOfThisFeature, featureIndex);
        newNode.left = buildDecisionTree(leftSubTree, maxDepth - 1);
        newNode.right = buildDecisionTree(rightSubtree, maxDepth - 1);

        return newNode;
    }

    private int getUniqueFeatureToSplitOn(List<Instance> instances, List<Integer> usedFeatures) {
        return new C45DecisionTreeTrainer().getFeatureWithLeastEntropy(instances, usedFeatures);
    }

    private void removeFeatureFromFeatureVector(List<Instance> instances, int featureIndex) {
        //for(Instance instance: instances)
          //  instance.getFeatureVector().getFeatureVectorKeys().remove(featureIndex);
        usedFeatures.add(featureIndex);
    }

    private Label populateMajorityLabel(List<Instance> instances) {
        Predictor majorityClassifier = new MajorityClassifier();
        majorityClassifier.train(instances);
        return majorityClassifier.predict(null);
    }

    private Label predictLabelWhenAllLabelsAreEqual(List<Instance> instances, boolean allLabelsEqual) {
        Label predictLabel = null;
        if(allLabelsEqual && instances.listIterator().hasNext())
            predictLabel = instances.listIterator().next().getLabel();
        return predictLabel;
    }

    private void divideFeatureVectorsBasedOnMean
            (List<Instance> instances, List<Instance> leftSubTree, List<Instance> rightSubtree, double mean, int featureIndex) {
        for (Instance instance : instances) {
            if (instance.getFeatureVector().get(featureIndex) != null && instance.getFeatureVector().get(featureIndex) <= mean)
                leftSubTree.add(instance);
            else
                rightSubtree.add(instance);
        }
    }

    private Double computeMeanForFeature(Integer featureIndex, List<Instance> instances) {
        Double sum = 0.0;
        for (Instance instance : instances){
            if(instance.getFeatureVector().get(featureIndex) != null)
                sum += instance.getFeatureVector().get(featureIndex);
        }
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
        for(Instance instance : instances)
            try {
                // Helps in computing majority label after a particular feature is removed from the data set
                trainingInstances.add((Instance)instance.clone());
            } catch (CloneNotSupportedException e) {
                e.printStackTrace();
            }
        rootNode = this.buildDecisionTree(instances, getMaxDepthArgument());
        cleanUp();
    }

    private void cleanUp() {
        trainingInstances.clear();
    }

    private int getMaxDepthArgument() {
        int max_decision_tree_depth = 4;
        if (CommandLineUtilities.hasArg("max_decision_tree_depth"))
            max_decision_tree_depth = CommandLineUtilities.getOptionValueAsInt("max_decision_tree_depth");
        return max_decision_tree_depth;
    }

    @Override
    public Label predict(Instance instance) {
        Node treeNode = rootNode;

        while (!treeNode.isLeaf && (treeNode.left != null || treeNode.right != null) ) {
            Double instanceFeatureValue = instance.getFeatureVector().get(treeNode.featureIndex);
            if(instanceFeatureValue == null) {
                return treeNode.prediction;
            }

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
            this.prediction = new ClassificationLabel(1);
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
