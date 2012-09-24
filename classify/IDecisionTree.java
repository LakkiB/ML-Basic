package cs475.classify;
import cs475.dataobject.Instance;

import java.util.HashMap;
import java.util.List;

public interface IDecisionTree {
    void construct(List<Instance> instances);
    void addNode(HashMap<Long, Double> node);
    void removeNode(HashMap<Long, Double> node);
}
