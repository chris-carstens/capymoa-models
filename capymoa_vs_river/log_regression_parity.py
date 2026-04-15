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


from capymoa.datasets import Electricity
from custom_capymoa.classifier import LogisticRegression as CapyLR
from river.linear_model import LogisticRegression as RiverLR
from river import optim
from capymoa.evaluation import prequential_evaluation

# Common Hyperparameters
seed = 42
learning_rate = 0.01
l2_regularization = 0.0
clip_gradient = 1e12
intercept_lr = 0.01

# 1. Re-initialize streams
stream_native = Electricity()

print("--- Initializing Models ---")

# CapyMOA Initialization
capy_model_native = CapyLR(
    schema=stream_native.get_schema(), 
    random_seed=seed, 
    learning_rate=learning_rate, 
    l2_regularization=l2_regularization,
    clip_gradient=clip_gradient,
    intercept_lr=intercept_lr,
)

print("Running CapyMOA Evaluation...")
tracemalloc.start()
start_native = time.perf_counter()
results = prequential_evaluation(stream=stream_native, learner=capy_model_native, max_instances=45312)
native_time = time.perf_counter() - start_native
_, capy_peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
capy_accuracy = results['cumulative'].accuracy()
capy_f1 = results['cumulative'].f1_score()
capy_precision = results['cumulative'].precision()
capy_recall = results['cumulative'].recall()

# 2. River Initialization and loop
stream_river = Electricity()
river_model_native = RiverLR(
    optimizer=optim.SGD(lr=learning_rate),
    l2=l2_regularization,
    intercept_lr=intercept_lr,
)

print("Running River Python Evaluation...")
tracemalloc.start()
start_river = time.perf_counter()
total_river = 0
river_correct = 0
river_y_true_list = []
river_y_pred_list = []
while stream_river.has_more_instances() and total_river < 45312:
    instance = stream_river.next_instance()
    x_dict = {i: val for i, val in enumerate(instance.x)}
    y_true = instance.y_index
    
    pred_river_raw = river_model_native.predict_one(x_dict)
    
    pred_river = 1 if pred_river_raw else 0
    if isinstance(pred_river_raw, int):
        pred_river = pred_river_raw
        
    river_y_true_list.append(y_true)
    river_y_pred_list.append(pred_river)
    if pred_river == y_true:
        river_correct += 1
        
    river_model_native.learn_one(x_dict, y_true)
    total_river += 1
river_time = time.perf_counter() - start_river
_, river_peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
river_accuracy = (river_correct / total_river) * 100
river_f1 = f1_score(river_y_true_list, river_y_pred_list, average='weighted') * 100
river_precision = precision_score(river_y_true_list, river_y_pred_list, average='weighted', zero_division=0) * 100
river_recall = recall_score(river_y_true_list, river_y_pred_list, average='weighted', zero_division=0) * 100


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
