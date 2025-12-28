# Quantum Neural Networks Regression Analysis

**Assessing the Advantages and Limitations of Quantum Neural Networks in Regression Tasks**

## Abstract

The development of quantum neural networks (QNNs) has attracted considerable attention due to their potential to surpass classical models in certain machine learning tasks. Nonetheless, it remains unclear under which conditions QNNs provide concrete benefits over classical neural networks (CNNs). This study addresses this question by performing both qualitative and quantitative analyses of classical and quantum models applied to regression problems, using two target functions with contrasting properties.  Additionally, the work explores the methodological difficulties inherent in making fair comparisons between QNNs and CNNs. 

The findings reveal a distinct advantage of QNNs in a specific quantum machine learning context. In particular, QNNs excelled at approximating the sinusoidal function, achieving errors up to seven orders of magnitude lower than their classical counterparts. However, their performance was limited in other cases, emphasizing that QNNs are highly effective for certain tasks but not universally superior. These results reinforce the principles of the "No Free Lunch" theorem, highlighting that no single model outperforms all others across every problem domain.

## Repository Structure

```
Quantum-Neural-Networks-in-Regression-Tasks/
│
├── Data/                           # Main data directory
│   ├── classical/                  # Classical neural network results
│   │   ├── Heaviside_comparing_by_layer/
│   │   ├── Heaviside_comparing_by_parameter/
│   │   ├── sino_comparing_by_layer/
│   │   └── sino_comparing_by_parameter/
│   │
│   └── quantum/                    # Quantum neural network results
│       ├── 1layer/                 # 100 experiments with 1 layer
│       ├── 2layer/                 # 100 experiments with 2 layers
│       ├── 3layer/                 # 100 experiments with 3 layers
│       ├── 4layer/                 # 100 experiments with 4 layers
│       └── 5layer/                 # 100 experiments with 5 layers
│
├── ClassicalModel.py               # Classical neural network implementation
├── gates.py                        # Quantum gates definitions
├── models_parallel.py              # Quantum model definition and training helpers
├── run_parallel_quantum.py         # Quantum experiments runner
├── Figure_2_and_4.ipynb            # Figure generation (paper figures 2 and 4)
├── Figure_3_and_5.ipynb            # Figure generation (paper figures 3 and 5)
├── Find_best_results.ipynb         # Sweep to locate best-performing runs
└── README.md                       
```

## Dataset Description

### Classical Data (`Data/classical/`)
- **Functions tested**: Heaviside and Sinusoidal (sino)
- **Comparisons**: By layer count and by neuron count per layer
- **Experiments**: 100 seeds per activation function configuration
- **Purpose**: Establish classical baseline performance

### Quantum Data (`Data/quantum/`)
- **Architecture variations**: 5 different layer configurations
- **Experiments**:  100 runs per configuration
- **Purpose**:  Evaluate quantum model performance across different architectures

## Key Findings

✅ **QNN Advantages:**
- Superior performance on sinusoidal function approximation
- Errors up to **7 orders of magnitude lower** than classical models in specific contexts

⚠️ **QNN Limitations:**
- Performance varies significantly by task
- Not universally superior to classical approaches
- Context-dependent effectiveness

## Scripts Overview

- **`ClassicalModel.py`**: Implements classical neural network architectures for regression tasks
- **`gates.py`**: Defines quantum gates used in QNN circuits
- **`models_parallel.py`**: Defines the `Quantum_Network` class and helpers for training/evaluating quantum models
- **`run_parallel_quantum.py`**: Orchestrates quantum model experiments
- **`Figure_2_and_4.ipynb`**: Generates the paper-ready figures 2 and 4
- **`Figure_3_and_5.ipynb`**: Generates the paper-ready figures 3 and 5
- **`Find_best_results.ipynb`**: Searches trained runs to surface best-performing configurations




## Citation

If you use this code or data in your research, please cite: 

```bibtex
@article{de2025assessing,
  title={Assessing the Advantages and Limitations of Quantum Neural Networks in Regression Tasks},
  author={de Limaa, Gubio G and Farias, Tiago de S and Ricardo, Alexandre C and Boas, Celso Jorge Villa},
  journal={arXiv preprint arXiv:2509.00854},
  year={2025}
}
```
---
