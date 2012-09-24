package cs475.dataobject;

import cs475.dataobject.*;
import cs475.dataobject.label.Label;

import java.io.Serializable;

public class Instance implements Serializable {

	Label _label = null;
	cs475.dataobject.FeatureVector _feature_vector = null;

	public Instance(cs475.dataobject.FeatureVector feature_vector, Label label) {
		this._feature_vector = feature_vector;
		this._label = label;
	}

	public Label getLabel() {
		return _label;
	}

	public void setLabel(Label label) {
		this._label = label;
	}

	public cs475.dataobject.FeatureVector getFeatureVector() {
		return _feature_vector;
	}

	public void setFeatureVector(cs475.dataobject.FeatureVector feature_vector) {
		this._feature_vector = feature_vector;
	}
	
	
}
