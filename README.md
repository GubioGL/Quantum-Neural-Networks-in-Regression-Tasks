# Quantum Neural Networks Regression Analysis

**Assessing the Advantages and Limitations of Quantum Neural Networks in Regression Tasks**

## Abstract

The development of quantum neural networks (QNNs) has attracted considerable attention due to their potential to surpass classical models in certain machine learning tasks. Nonetheless, it remains unclear under which conditions QNNs provide concrete benefits over classical neural networks (CNNs). This study addresses this question by performing both qualitative and quantitative analyses of classical and quantum models applied to regression problems, using two target functions with contrasting properties.  Additionally, the work explores the methodological difficulties inherent in making fair comparisons between QNNs and CNNs. 

The findings reveal a distinct advantage of QNNs in a specific quantum machine learning context. In particular, QNNs excelled at approximating the sinusoidal function, achieving errors up to seven orders of magnitude lower than their classical counterparts. However, their performance was limited in other cases, emphasizing that QNNs are highly effective for certain tasks but not universally superior. These results reinforce the principles of the "No Free Lunch" theorem, highlighting that no single model outperforms all others across every problem domain.

## Repository Structure

```
Quantum-Neural-Networks-in-Regression-Tasks/
│
├── data_regression/          # Main data directory
│   ├── classica/             # Classical neural network results
│   │   ├── heaviside/        # Heaviside function experiments
│   │   │   ├── por_layer/    # Comparison by layer
│   │   │   └── por_camada/   # Comparison by neuron count
│   │   └── sino/             # Sinusoidal function experiments
│   │       ├── por_layer/    # Comparison by layer
│   │       └── por_camada/   # Comparison by neuron count
│   │
│   └── quantica/             # Quantum neural network results
│       ├── layer_1/          # 100 experiments with 1 layer
│       ├── layer_2/          # 100 experiments with 2 layers
│       ├── layer_3/          # 100 experiments with 3 layers
│       ├── layer_4/          # 100 experiments with 4 layers
│       └── layer_5/          # 100 experiments with 5 layers
│
├── ClassicalModel.py         # Classical neural network implementation
├── gates.py                  # Quantum gates definitions
├── models_parallel.py        # Parallel model execution
├── run_parallel_quantum.py   # Quantum experiments runner
├── plots-predição.py         # Prediction plotting script
├── plot-poster-1.ipynb       # Poster visualization notebook
├── plots-comparação. ipynb    # Comparison analysis notebook
└── README.md                 # This file
```

## Dataset Description

### Classical Data (`data_regression/classica/`)
- **Functions tested**: Heaviside and Sinusoidal (sino)
- **Comparisons**: By layer count and by neuron count per layer
- **Experiments**: 100 seeds per activation function configuration
- **Purpose**: Establish classical baseline performance

### Quantum Data (`data_regression/quantica/`)
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
- **`models_parallel.py`**: Enables parallel execution of multiple model configurations
- **`run_parallel_quantum.py`**: Orchestrates quantum model experiments
- **`plots-predição.py`**: Generates prediction visualizations
- **`plot-poster-1.ipynb`**: Creates publication-ready figures for posters
- **`plots-comparação.ipynb`**: Analyzes and compares classical vs quantum performance

## Getting Started

### Prerequisites
```bash
# Python dependencies (example)
pip install numpy scipy matplotlib jupyter
pip install qiskit pennylane torch
```

### Running Experiments

1. **Classical experiments:**
```bash
python ClassicalModel.py
```

2. **Quantum experiments:**
```bash
python run_parallel_quantum.py
```

3. **Generate visualizations:**
```bash
python plots-predição.py
# Or open Jupyter notebooks
jupyter notebook plots-comparação.ipynb
```

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

## License

[Add your preferred license here - MIT, Apache 2.0, etc.]

## Contact

For questions or collaborations, please open an issue or contact [your email].

---

**Note**: This research demonstrates that while QNNs show promise in specific quantum machine learning contexts, careful consideration must be given to the problem domain. The "No Free Lunch" theorem remains relevant in the quantum computing era. 
