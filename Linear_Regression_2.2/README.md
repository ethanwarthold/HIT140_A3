# Linear Regression 2.2
  This project builds and trains a multiple linear regression model to predict the number of goals scored by a team in a 2026 FIFA World Cup match.  
  Metrics such as mean absolute error, mean squared error, root mean squared error, R squared, and more are reported.  
  Other types of regression models are used to compare with linear regression, those being: Ada Boost, Bagging, Gradient Boost, and Random Forest.

### Dependences
  python 3.10+  

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

### Project Files
  visualise.py - a light wrapper over seaborn to quickly display histograms or scatterplots with trendlines for the explanatory variables.  

  transform.py - contains functions for applying transforms to the data.  

  linreg_sm.py - uses statsmodels multiple linear regression to train a model and print metrics, this is used to test optimisation techniques.  

  regression.py - contains functions for all scikit-learn regression models used, each function prints the error metrics MAE, MSE, RMSE, Norm. RMSE, and R squared.  

  main.py - entry point, uses functions in the other files in a procedural manner.  

  data/wc_data.py - match the formatting of the World Cup matches CSV to match that of the full international matches dataset.  

  data/main.py - collect explanatory variables from the data in a reproducible manner.  

  data/Data2.2.csv - the final dataset used for regression.

### Sources
  FIFA points data:
  https://inside.fifa.com/fifa-rankings/world-ranking/men?dateId=FRS_Male_Football_20260401

  Previous match dates and scoring data:
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
