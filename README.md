# Wrapping Linear Models from River to CapyMOA

## 🎯 Objective

Integrate additional linear models from the River library into CapyMOA. The objective is to provide high-performance, Java-backed implementations (MOA) that are mathematically identical to their River counterparts.

**Current Status:**
- ✅ **Logistic Regression** (Binary Classification) - Successfully integrated and verified.
- ✅ **Softmax Regression** (Multi-class Classification) - Successfully integrated and verified.

## 🧠 Background Context

*   **River:** A Python-native library for online machine learning. It offers many linear models but can be slower due to Python-native execution.
*   **CapyMOA:** A framework leveraging the Java-based MOA (Massive Online Analysis) backend, offering significantly faster performance for streaming data.
*   **The Mission:** Bridge the gap by implementing River's linear model logic in Java (MOA) and wrapping it for CapyMOA, ensuring exact numerical parity.

---

## 🚀 Accuracy Parity & Verification

We verify our implementations by comparing them against River step-by-step. Our goal is **numerical parity**: given the same data and hyperparameters, both models should produce identical results.

### 1. Logistic Regression Parity
Identical accuracy achieved on the **Electricity (Elec2)** dataset (45,312 instances).
- **Metric:** Accuracy matching to within 0.01%.
- **Verification Script:** `capymoa_vs_river/log_regression_parity.py`

### 2. Softmax Regression Parity
Identical accuracy achieved on the **Covtype** dataset (multi-class).
- **Metric:** Accuracy matching verified through comparative evaluation.
- **Verification Script:** `capymoa_vs_river/softmax_parity.py`

---

## 🛠 Usage Instructions

### Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

### Running Parity Checks

You can run the parity scripts directly to verify the implementations:

```bash
# Verify Logistic Regression
python3 capymoa_vs_river/log_regression_parity.py

# Verify Softmax Regression
python3 capymoa_vs_river/softmax_parity.py
```

### Compiling Custom Java Classifiers

If you modify the Java source code in `custom_moa/`, you must recompile the classes:

**Command:**
```bash
# From the project root
javac -cp .venv/lib/python3.11/site-packages/capymoa/jar/moa.jar -d custom_moa/ custom_moa/moa/classifiers/functions/*.java
```

> [!IMPORTANT]
> If you are using a different Python version, update the classpath `-cp` to point to the correct `moa.jar` location in your `.venv`.

---

## 📂 Project Structure

```
sda-project/
├── README.md
├── requirements.txt                # Python dependencies
├── .venv/                         # Virtual environment
├── custom_moa/                    # Java Implementations (MOA)
│   └── moa/classifiers/functions/
│       ├── LogisticRegression.java
│       └── SoftmaxRegression.java
├── custom_capymoa/                 # Python Wrappers (CapyMOA)
│   └── classifier/
│       ├── _logistic_regression.py
│       └── _softmax_regression.py
├── capymoa_vs_river/              # Verification & Parity Scripts
│   ├── log_regression_parity.py
│   ├── softmax_parity.py
│   └── data/                      # Datasets (Electricity, Covtype)
└── tests/                         # Unit tests
```

---

## 🤖 AI Agent Technical Workflow

If you are an AI agent assisting with this project, follow this process for wrapping new models:

1.  **Reference River:** Study the [River source code](https://github.com/online-ml/river) for the model's logic.
2.  **Java Backend:** Implement the logic in Java within `custom_moa/`, extending appropriate MOA base classes.
3.  **Python Wrapper:** Create the CapyMOA wrapper in `custom_capymoa/classifier/`.
4.  **Parity Verification:** Create a script in `capymoa_vs_river/` to compare CapyMOA and River outputs at every step.


