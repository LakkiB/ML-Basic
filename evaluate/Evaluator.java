package cs475.evaluate;

import cs475.dataobject.Instance;
import cs475.classifier.Predictor;

import java.util.List;

public abstract class Evaluator {

	public abstract double evaluate(List<Instance> instances, Predictor predictor);
}
