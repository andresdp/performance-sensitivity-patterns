# Performance Sensitivity Analysis of Microservices Patterns
This is a reproducibility kit for a paper "Data-driven Understanding of Design Decisions
in Pattern-based Microservices Architectures" submitted to ECSA 2025.

A performance sensitivity analysis is implemented for the following microservices patterns:
* *Gateway Offloading (GO)*
* *Anti-corruption Layer (ACL)*
* *Command Query Responsibility Segreggation (CQRS)*

These patterns are taken from an existing work by [Pinciroli et al.](https://cs.gssi.it/catia.trubiani/download/2023-ICSA-DesignPatterns-Performance.pdf) 
The authors' original dataset can be accessed [here](https://zenodo.org/records/7524410).


## Usage
The analysis of each pattern, along with the necessary notebooks anda data, is located in the corresponding folder.
The main Jupyter notebook for a running a given pattern analysis is called *robustness-paper.ipynb*. This notebook relies on a set of common classes and utility functions defined in the *archspace.py* file.

The creation of a Python environment (e.g., with Conda) is recommended. A *requirements.txt* file is provided.

Part of the data analysis and the PRIM algorithm are based on the [EMA Workbench](https://github.com/quaquel/EMAworkbench) library for explorative modeling and analysis.

The CART algorithm is adapted from the [Scikit-learn](https://scikit-learn.org/stable/modules/tree.html) library.
