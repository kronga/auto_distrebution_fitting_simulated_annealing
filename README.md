# Automatic Distribution Fitting of Tabular Features using Simulated Annealling

ADD HERE A 2-sentence summary later.

## Summery
Take the abstract from the summary later.

## Example
Add an example as shown in the report.

## Sample data
ADD here later

## Structure
The project is structured as follows:
1. **main.py** - A demo file of the main task of the project.
2. **dists.py** - A list of distributions and their properties.
3. **experiments.py** - A list of experiments presented in the project's report.
4. **ploter.py** - A helper class to plot the graphs shown in the project's report .
5. **algo.py** -  An algorithm that run on each column in a DF and allocates the best distribution (with parameters) for it using the simulated annealing algorithm.
6. **algo_state.py** -  A helper class holding a dist and its parameters to represent a state in the distribution space.
7. **smart_gap_fillter.py** - A KNN-based feature-oriented filler, taking the similar top 'k' rows without some feature to each row with this data and fill it.


## Getting started
1. Clone the repo
2. Install the 'requirements.txt' file (pip install requirements.txt)
3. Enter a "data.csv" file to the "data" folder in the project's root.
4. Run the project from the 'main.py' file (python main.py or python3 main.py from the terminal)

## Dependencies
- Python               3.7.1
- numpy                1.20.2
- matplotlib           3.4.0
- pandas               1.2.3
- seaborn              0.11.1
- scikit-learn         1.0.1
- scipy                1.7.3

These can be found in the **requirements.txt** and easily installed using the "pip install requirements.txt" command in your terminal. 