# BI_tenserflow
Python script that uses Keras/tenserflow to create, examine, and train an ANN that can determine which members are likely to leave a bank

# Prep - windows
1. Install Anaconda distr and create a new env with Anaconda/Python
   - `conda create -n tenserflow python=3.5 anaconda`
2. Activate the env
   - `activate tenserflow`
3. Install deps
   - `conda install theano`
   - `conda install mingw libpython`
   - `pip install tenserflow`
   - `pip install keras`
4. Update Pakages
   - `conda update -all`

## Basic usage
### Always run `ann.py` in your Anaconda console to build/encode the bank data before running any of the scripts in `app` dir
- `tweak.py` 
  - running this starts a series of tests via `GridSearchCV` and `KerasClassifier`. You can set your own params inside `param_grid`
- `eval.py`
  - running this just runs a single test on the set and sets variables to the mean and varience of the results
