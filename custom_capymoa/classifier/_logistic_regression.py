from __future__ import annotations
import typing

from capymoa.base import MOAClassifier
from capymoa.stream import Schema

# Import the Java MOA class via the JPype bridge
import moa.classifiers.functions as moa_functions


class LogisticRegression(MOAClassifier):
    """Online Logistic Regression classifier using SGD with log loss.

    This classifier performs binary classification by minimizing the log loss
    (binary cross-entropy) using Stochastic Gradient Descent (SGD). It applies
    a sigmoid activation function to produce class probabilities.

    >>> from capymoa.classifier import LogisticRegression
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = LogisticRegression(stream.get_schema(), learning_rate=0.01)
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")

    Notes
    -----
    - For best results, scale/normalize features before training (e.g., using
      a preprocessing.StandardScaler in the pipeline).
    - The bias/intercept is updated separately without regularization.
    """

    def __init__(
        self,
        schema: typing.Union[Schema, None] = None,
        random_seed: int = 0,
        learning_rate: float = 0.01,
        intercept_lr: float = None,
        l2_regularization: float = 0.0,
        clip_gradient: float = 1e12,
    ):
        """Construct the Logistic Regression classifier.

        :param schema: Describes the datastream's structure.
        :param random_seed: Seed for reproducibility. Note: this classifier is
        deterministic and does not use randomness.
        :param learning_rate: The step size for SGD weight updates.
        :param intercept_lr: The step size for bias/intercept updates.
        :param l2_regularization: L2 penalty coefficient (weight decay).
        :param clip_gradient: The maximum absolute value for gradient clipping.
        """
        if intercept_lr is None:
            intercept_lr = learning_rate

        moa_learner = moa_functions.LogisticRegression()

        # Configure hyperparameters
        moa_learner.learningRateOption.setValue(learning_rate)
        moa_learner.interceptLearningRateOption.setValue(intercept_lr)
        moa_learner.l2RegularizationOption.setValue(l2_regularization)
        moa_learner.clipGradientOption.setValue(clip_gradient)

        # Initialize the parent MOAClassifier
        super(LogisticRegression, self).__init__(
            moa_learner=moa_learner,
            schema=schema,
            random_seed=random_seed,
        )

    def __str__(self):
        return "LogisticRegression CapyMOA Classifier"
