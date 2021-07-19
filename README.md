# reward-weighted-regression

This repository contains the source code for the experiments presented in *Reward-Weighted Regression Converges to a Global Optimum* by Miroslav Štrupl, Francesco Faccio, Dylan R. Ashley, Rupesh Kumar Srivastava, and Jürgen Schmidhuber.

To produce the plots shown in the paper, first ensure you have python 3.8.0 installed and execute the following in a bash shell:
```bash
pip install -r requirements.txt
./build.sh
./tasks_0.sh
./plot.py results.pdf
```
After execution has completed, a `results.pdf` file should have been generated.
