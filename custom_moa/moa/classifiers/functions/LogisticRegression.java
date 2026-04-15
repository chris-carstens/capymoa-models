/*
 *    LogisticRegression.java
 *
 *    Logistic Regression classifier for online/streaming learning.
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

package moa.classifiers.functions;

import moa.classifiers.AbstractClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.StringUtils;

import com.github.javacliparser.FloatOption;

import com.yahoo.labs.samoa.instances.Instance;

/**
 * Logistic Regression classifier for online/streaming learning.
 *
 * <p>
 * This classifier uses Stochastic Gradient Descent (SGD) to optimize
 * the log loss (binary cross-entropy). It applies a sigmoid activation
 * function to produce class probabilities.
 * </p>
 *
 * <h3>Algorithm Overview</h3>
 * <ul>
 * <li><b>Forward pass:</b> z = w·x + bias; p = σ(z) = 1/(1+exp(-z))</li>
 * <li><b>Loss:</b> Log loss = -[y·log(p) + (1-y)·log(1-p)]</li>
 * <li><b>Gradient:</b> ∇L = (p - y)·x (for weights), (p - y) (for bias)</li>
 * <li><b>Update:</b> w ← w·(1 - lr·λ) - lr·∇L (with L2 weight decay)</li>
 * </ul>
 *
 * <h3>Parameters</h3>
 * <ul>
 * <li><b>learningRate</b> (-r): The step size for SGD updates. Default:
 * 0.01</li>
 * <li><b>l2Regularization</b> (-l): L2 penalty coefficient (weight decay).
 * Default: 0.0</li>
 * <li><b>clipGradient</b> (-c): Maximum absolute value for gradient clipping.
 * Default: 1e12</li>
 * </ul>
 */
public class LogisticRegression extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Logistic Regression for online binary classification using SGD with log loss.";
    }

    // ---------------------------------------------------------------
    // MOA Options (configurable via CLI / GUI)
    // ---------------------------------------------------------------

    public FloatOption learningRateOption = new FloatOption(
            "learningRate", 'r',
            "Learning rate for SGD weight updates.",
            0.01, 0.0, Double.MAX_VALUE);

    public FloatOption interceptLearningRateOption = new FloatOption(
            "interceptLearningRate", 'i',
            "Learning rate for the bias / intercept term.",
            0.01, 0.0, Double.MAX_VALUE);

    public FloatOption l2RegularizationOption = new FloatOption(
            "l2Regularization", 'l',
            "L2 regularization parameter (weight decay). Pushes weights towards 0.",
            0.0, 0.0, Double.MAX_VALUE);

    public FloatOption clipGradientOption = new FloatOption(
            "clipGradient", 'c',
            "Clips the absolute value of the gradient to this maximum.",
            1e12, 0.0, Double.MAX_VALUE);

    // ---------------------------------------------------------------
    // Model state
    // ---------------------------------------------------------------

    /**
     * Weight vector. Index i corresponds to attribute i
     */
    protected DoubleVector weights;

    /** Bias / intercept term, updated separately. */
    protected double bias;

    /** Cached hyperparameters (read from options at reset). */
    protected double learningRate;
    protected double interceptLearningRate;
    protected double l2Regularization;
    protected double clipGradient;

    // ---------------------------------------------------------------
    // Core methods
    // ---------------------------------------------------------------

    /**
     * Sigmoid activation function.
     *
     * @param z the raw dot-product (logit)
     * @return σ(z) ∈ (0, 1)
     */
    protected static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        } else {
            double expZ = Math.exp(z);
            return expZ / (1.0 + expZ);
        }
    }

    /**
     * Computes the dot product between an instance's feature values and
     * the current weight vector, adding the bias.
     *
     * @param inst the instance
     * @return w·x + bias
     */
    protected double rawDot(Instance inst) {
        double result = 0.0;
        for (int i = 0; i < inst.numValues(); i++) {
            int idx = inst.index(i);
            if (idx != inst.classIndex() && !inst.isMissingSparse(i)) {
                if (idx < this.weights.numValues()) {
                    result += inst.valueSparse(i) * this.weights.getValue(idx);
                }
                // else weight is implicitly 0 (not yet allocated)
            }
        }
        return result + this.bias;
    }

    /**
     * Resets the model to its initial state.
     */
    @Override
    public void resetLearningImpl() {
        this.weights = new DoubleVector();
        this.bias = 0.0;

        // Read hyperparameters from option objects
        this.learningRate = this.learningRateOption.getValue();
        this.interceptLearningRate = this.interceptLearningRateOption.getValue();
        this.l2Regularization = this.l2RegularizationOption.getValue();
        this.clipGradient = this.clipGradientOption.getValue();
    }

    /**
     * Trains the model on a single instance using SGD with log loss.
     *
     * <p>
     * Steps:
     * <ol>
     * <li>Compute raw logit z = w·x + bias</li>
     * <li>Compute predicted probability p = σ(z)</li>
     * <li>Compute loss gradient = p - y (where y ∈ {0, 1})</li>
     * <li>Clip gradient</li>
     * <li>Apply L2 weight decay</li>
     * <li>Update weights: w_i ← w_i - lr · gradient · x_i</li>
     * <li>Update bias: bias ← bias - lr · gradient</li>
     * </ol>
     *
     * @param inst the training instance
     */
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (inst.classIsMissing()) {
            return;
        }

        // Target class value (0 or 1)
        double y = inst.classValue();

        // Forward pass: compute predicted probability
        double z = this.rawDot(inst);
        double p = sigmoid(z);

        // Compute loss gradient: dL/dz = p - y
        double lossGradient = (p - y) * inst.weight();

        // Clip gradient
        lossGradient = Math.max(-this.clipGradient, Math.min(this.clipGradient, lossGradient));

        // Apply L2 regularization (weight decay) and gradient update for each feature
        for (int i = 0; i < inst.numValues(); i++) {
            int idx = inst.index(i);
            if (idx != inst.classIndex() && !inst.isMissingSparse(i)) {
                double xi = inst.valueSparse(i);

                // Current weight (may be 0 if not yet allocated)
                double currentWeight = this.weights.getValue(idx);

                // L2 regularization: decay existing weight
                if (this.l2Regularization > 0.0) {
                    currentWeight *= (1.0 - this.learningRate * this.l2Regularization);
                }

                // SGD update: w -= lr * gradient * x_i
                currentWeight -= this.learningRate * lossGradient * xi;

                this.weights.setValue(idx, currentWeight);
            }
        }

        // Update bias (no regularization applied to bias)
        this.bias -= this.interceptLearningRate * lossGradient;
    }

    /**
     * Returns the class probability distribution for a given instance.
     *
     * <p>
     * For binary classification, returns [P(class=0), P(class=1)].
     * </p>
     *
     * @param inst the instance to classify
     * @return double array of class probabilities
     */
    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.weights == null) {
            // Model not yet trained — return uniform distribution
            return new double[inst.numClasses()];
        }

        double z = this.rawDot(inst);
        double pClass1 = sigmoid(z);
        double pClass0 = 1.0 - pClass1;

        if (inst.numClasses() == 2) {
            return new double[] { pClass0, pClass1 };
        }

        // Fallback for unexpected multi-class (shouldn't happen with binary LR)
        double[] votes = new double[inst.numClasses()];
        votes[0] = pClass0;
        if (votes.length > 1) {
            votes[1] = pClass1;
        }
        return votes;
    }

    // ---------------------------------------------------------------
    // Reporting / description methods
    // ---------------------------------------------------------------

    @Override
    public void getModelDescription(StringBuilder result, int indent) {
        StringUtils.appendIndented(result, indent, toString());
        StringUtils.appendNewline(result);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[] {
                new Measurement("model bias", this.bias),
                new Measurement("num weights", this.weights != null ? this.weights.numValues() : 0)
        };
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    /**
     * Returns a human-readable description of the model state.
     */
    @Override
    public String toString() {
        if (this.weights == null || this.weights.numValues() == 0) {
            return "LogisticRegression: No model built yet.\n";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("LogisticRegression (Log Loss / Sigmoid)\n");
        sb.append("  Learning Rate: ").append(this.learningRate).append("\n");
        sb.append("  L2 Regularization: ").append(this.l2Regularization).append("\n");
        sb.append("  Gradient Clip: ").append(this.clipGradient).append("\n");
        sb.append("  Bias: ").append(String.format("%.6f", this.bias)).append("\n");
        sb.append("  Weights (").append(this.weights.numValues()).append(" features):\n");

        int maxDisplay = Math.min(this.weights.numValues(), 20);
        for (int i = 0; i < maxDisplay; i++) {
            sb.append("    w[").append(i).append("] = ")
                    .append(String.format("%.6f", this.weights.getValue(i))).append("\n");
        }
        if (this.weights.numValues() > maxDisplay) {
            sb.append("    ... (").append(this.weights.numValues() - maxDisplay).append(" more)\n");
        }

        return sb.toString();
    }
}
