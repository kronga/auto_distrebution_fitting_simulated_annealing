# Automatic Distribution Fitting of Tabular Features using Simulated Annealling
Allocating a parametric distribution to features in a tumbler data is a well-known problem for all data analysis as this provides a wide range of abilities such as better explainability, generation of new samples, and better picking of classification and regressions model using meta-learning. In this study, we propose a simulated annealing approach to allocate a parametric distribution for features, obtaining the optimal distribution and its parameters. 

## Usage Example
2. Run the **main.py** script.
3. Chcekout the results in the "/results" folder

## Structure
The project is structured as follows:
1. **main.py** - A demo file of the main task of the project.
2. **dists.py** - A list of distributions and their properties.
3. **experiments.py** - A list of experiments presented in the project's report.
4. **algo.py** -  An algorithm that run on each column in a DF and allocates the best distribution (with parameters) for it using the simulated annealing algorithm.
5. **algo_state.py** -  A helper class holding a dist and its parameters to represent a state in the distribution space.
6. **naive.py** - A brute-force algorithm used for comparison for fitting the best PDF and its parameters for each column in the dataset.
7. **smart_gap_fillter.py** - A KNN-based feature-oriented filler, taking the similar top 'k' rows without some feature to each row with this data and fill it.


## Getting started
1. Clone the repo
2. Create a folder "data" and a folder "results" inside the project's folder.
3. Run the project from the 'main.py' file (python main.py or python3 main.py from the terminal)

## Dependencies
- Python               3.7.1
- numpy                1.20.2
- matplotlib           3.4.0
- pandas               1.2.3
- scipy                1.7.3

