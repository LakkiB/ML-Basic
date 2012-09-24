package cs475.utils;

import cs475.dataobject.label.Label;

import java.io.*;

public class PredictionsWriter {

	private Writer _writer;

	public PredictionsWriter(String prediction_file) throws IOException {
		this._writer = new BufferedWriter(new FileWriter(prediction_file));
	}

	public void close() throws IOException {
		this._writer.close();
	}
	
	public void writePrediction(Label label) throws IOException {
		this._writer.write(label.toString());
		this._writer.write("\n");
	}
	
}
