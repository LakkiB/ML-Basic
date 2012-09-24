package cs475.classify;

import cs475.dataobject.Instance;
import cs475.dataobject.label.ClassificationLabel;
import cs475.dataobject.label.Label;

import java.io.Serializable;
import java.util.*;

public class DecisionTree extends Predictor{

    private Node rootNode;
    private List<Instance> trainingInstances;
    private Label majorityLabel;

    public DecisionTree() {
        rootNode = null;
        majorityLabel = null;
        trainingInstances = null;
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
        int featureIndex        = new C45DecisionTreeTrainer().getFeatureWithLeastEntropy(instances);
        double meanOfThisFeature    = computeMeanForFeature(featureIndex, instances);

        divideFeatureVectorsBasedOnMean(instances, leftSubTree, rightSubtree, meanOfThisFeature, featureIndex);

        // No way to split?
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
        rootNode = this.buildDecisionTree(instances, 4);
    }

    @Override
    public Label predict(Instance instance) {
        Node treeNodeD = rootNode;

        while (!treeNodeD.isLeaf && (treeNodeD.left != null || treeNodeD.right != null) ) {
            if (instance.getFeatureVector().get(treeNodeD.rootFeatureIndex) <= treeNodeD.mean)
                treeNodeD = treeNodeD.left;
            else
                treeNodeD = treeNodeD.right;
        }
        return  treeNodeD.prediction;
    }


    class Node implements Serializable {
        private Double mean;
        private Integer rootFeatureIndex;
        private boolean isLeaf;
        private Label prediction;
        Node left;
        Node right;


        public Node(Double mean, Integer featureIndex) {
            this.mean = mean;
            this.rootFeatureIndex = featureIndex;
            this.isLeaf = false;
            this.prediction = null;
        }

        public Node() {
            this.isLeaf = false;
            this.prediction = null;
        }

        public Node(Label majorityLabel, boolean isLeaf) {
            this.isLeaf = isLeaf;
            this.prediction = majorityLabel;

            // why am I doing this?
            this.mean = 99999999.0;
            this.rootFeatureIndex = -999999999;
        }
    }

   /* class LeafNode implements Node{
        Label prediction;

        public LeafNode(Label predictLabel) {
            prediction = predictLabel;
        }
    }*/
}
