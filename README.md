# Overview
This repository contains code for the paper:

Bryan Wilder, Bistra Dilkina, Milind Tambe. Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization. AAAI Conference on Artificial Intelligence. 2019.
```
@inproceedings{wilder2019melding,
 author = {Wilder, Bryan},
 title = {Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization},
 booktitle = {Proceedings of the 33rd AAAI Conference on Artificial Intelligence},
 year = {2019}
}
```

Included are differentiable solvers for LPs and submodular maximization, along with code to run the experiments in the paper. You can [download](https://bryanwilder.github.io/files/data_decisions_benchmarks.zip) the datasets from my website.

UPDATE: For the cora experiments, you can find the features [here](https://drive.google.com/file/d/1WWLgq552YJy_1HUw0GqyXO34k1-OYJxS/view?usp=sharing) and the graph structures [here](https://drive.google.com/file/d/1WWLgq552YJy_1HUw0GqyXO34k1-OYJxS/view?usp=sharing).

# Dependencies
* The linear programming experiments use the [Gurobi](http://www.gurobi.com/) solver.
* All code in the directory qpthlocal is derived from the [qpth](https://github.com/locuslab/qpth) library. It has been modified to support use of the Gurobi solver in the forward pass.
