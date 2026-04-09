
import sys
import os
import jpype
import time
import numpy as np

# Add the project root to sys.path
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

# Ensure custom Java classes are in the classpath
java_classes_root = os.path.join(project_root, "custom_moa")
if not jpype.isJVMStarted():
    jpype.addClassPath(java_classes_root)

from capymoa.datasets import Covtype
from custom_capymoa.classifier import SoftmaxRegression as CapySR
from river.linear_model import SoftmaxRegression as RiverSR
from river import optim, compose, preprocessing
from capymoa.evaluation import prequential_evaluation

# Initialize stream
stream_capy = Covtype()
max_instances = 10000

# Common Hyperparameters
seed = 42
learning_rate = 0.01
l2_regularization = 0.0001

print("--- Initializing Models ---")
# 1. CapyMOA Model
capy_model = CapySR(
    schema=stream_capy.get_schema(), 
    random_seed=seed, 
    learning_rate=learning_rate, 
    l2_regularization=l2_regularization,
)

# 2. River Model (bias added manually in the loop)
optimizer = optim.SGD(lr=learning_rate)
river_model = RiverSR(
    optimizer=optimizer,
    l2=l2_regularization,
)

print("Running CapyMOA Evaluation...")
start_native = time.perf_counter()
results = prequential_evaluation(stream=stream_capy, learner=capy_model, max_instances=max_instances)
native_time = time.perf_counter() - start_native
capy_accuracy = results['cumulative'].accuracy()

print("Running River Evaluation (manual bias added)...")
stream_river = Covtype()
start_river = time.perf_counter()
total_river = 0
river_correct = 0
while stream_river.has_more_instances() and total_river < max_instances:
    instance = stream_river.next_instance()
    # Convert to feature dictionary for River and add bias
    x_dict = {i: val for i, val in enumerate(instance.x)}
    x_dict['bias'] = 1.0
    y_true = instance.y_index
    
    # Test-then-train
    pred_river = river_model.predict_one(x_dict)
    if pred_river == y_true:
        river_correct += 1
        
    river_model.learn_one(x_dict, y_true)
    total_river += 1

river_time = time.perf_counter() - start_river
river_accuracy = (river_correct / total_river) * 100

print("\n--- Execution Summary ---")
print(f"CapyMOA Accuracy: {capy_accuracy:.2f}%")
print(f"River Accuracy:   {river_accuracy:.2f}%")
print(f"\nTotal Native CapyMOA Time:  {native_time:.4f} seconds")
print(f"Total Native River Time:    {river_time:.4f} seconds")
