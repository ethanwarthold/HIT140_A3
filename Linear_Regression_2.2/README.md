# Linear Regression 2.2
  This project builds and trains a multiple linear regression model to predict the number of goals scored by a team in a 2026 FIFA World Cup match.
  Metrics such as mean absolute error, mean squared error, root mean squared error, R squared, and more are reported.
  Other types of regression models are used to compare with linear regression, those being: Ada Boost, Bagging, Gradient Boost, and Random Forest.

### Dependences
  - matplotlib
  - numpy
  - pandas
  - scikit-learn
  - seaborn
  - statsmodels

  Install command with pip:
  ```
  pip install matplotlib numpy pandas scikit-learn seaborn statsmodels
  ```

### Usage
  From the Linear_Regression_2.2 directory:
  ```
  python main.py
  ```
  Alternatively, open main.py through a file explorer window.

### Project Structure
  visualise.py - a light wrapper over seaborn to quickly display histograms or scatterplots with trendlines for the explanatory variables.
  transform.py - contains functions for applying transforms to the data.
  linreg_sm.py - uses statsmodels multiple linear regression to train a model and print metrics, this is used to test optimisation techniques.
  regression.py - contains functions for all scikit-learn regression models used, each function prints the error metrics MAE, MSE, RMSE, Norm. RMSE, and R squared.
  main.py - entry point, uses functions in the other files in a procedural manner.

### Sources
  FIFA points data:
  https://inside.fifa.com/fifa-rankings/world-ranking/men?dateId=FRS_Male_Football_20260401

  Previous match dates and scoring data:
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
