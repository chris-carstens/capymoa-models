
import sys
import os
import jpype
import time
import tracemalloc
from sklearn.metrics import f1_score, precision_score, recall_score

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
max_instances = 40000

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
tracemalloc.start()
start_native = time.perf_counter()
results = prequential_evaluation(stream=stream_capy, learner=capy_model, max_instances=max_instances)
native_time = time.perf_counter() - start_native
_, capy_peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
capy_accuracy = results['cumulative'].accuracy()
capy_f1 = results['cumulative'].f1_score()
capy_precision = results['cumulative'].precision()
capy_recall = results['cumulative'].recall()

print("Running River Evaluation (manual bias added)...")
stream_river = Covtype()
tracemalloc.start()
start_river = time.perf_counter()
total_river = 0
river_correct = 0
river_y_true_list = []
river_y_pred_list = []
while stream_river.has_more_instances() and total_river < max_instances:
    instance = stream_river.next_instance()
    # Convert to feature dictionary for River and add bias
    x_dict = {i: val for i, val in enumerate(instance.x)}
    x_dict['bias'] = 1.0
    y_true = instance.y_index
    
    # Test-then-train
    pred_river = river_model.predict_one(x_dict)
    if pred_river is None:
        pred_river = -1  # default for unseen classes
    river_y_true_list.append(y_true)
    river_y_pred_list.append(pred_river)
    if pred_river == y_true:
        river_correct += 1
        
    river_model.learn_one(x_dict, y_true)
    total_river += 1

river_time = time.perf_counter() - start_river
_, river_peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
river_accuracy = (river_correct / total_river) * 100
river_f1 = f1_score(river_y_true_list, river_y_pred_list, average='macro') * 100
river_precision = precision_score(river_y_true_list, river_y_pred_list, average='macro', zero_division=0) * 100
river_recall = recall_score(river_y_true_list, river_y_pred_list, average='macro', zero_division=0) * 100

print("\n--- Execution Summary ---")
print(f"CapyMOA Accuracy:   {capy_accuracy:.2f}%")
print(f"CapyMOA F1:         {capy_f1:.2f}%")
print(f"CapyMOA Precision:  {capy_precision:.2f}%")
print(f"CapyMOA Recall:     {capy_recall:.2f}%")
print(f"\nRiver Accuracy:     {river_accuracy:.2f}%")
print(f"River F1:           {river_f1:.2f}%")
print(f"River Precision:    {river_precision:.2f}%")
print(f"River Recall:       {river_recall:.2f}%")
print(f"\nTotal Native CapyMOA Time:  {native_time:.4f} seconds")
print(f"CapyMOA Peak Memory:        {capy_peak_mem / 1024 / 1024:.4f} MB")
print(f"Total Native River Time:    {river_time:.4f} seconds")
print(f"River Peak Memory:          {river_peak_mem / 1024 / 1024:.4f} MB")
