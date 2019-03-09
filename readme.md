# ML Proj Tasks

## Process

 - Split Notebooks and take out repeated functions to seperate `.py` files - 70% done

## Data prep

### DONE

- contstruction year => binning
- scheme name
  - irrelevant - elimanated
- extraction type / extraction type group / extraction class - Do Nothing
- water quality vs quality group - Do Nothing
- region and region code - Do Nothing
- water point type / water point type group - Drop waterpoint_type_group

### Not tackled yet (Feature Engineering)
- amount_tsh
  - Lots os zeroes - doesn't make sense because if it's functional it can't have 0 "amount of water available to the water point"
  - Impute the `functional/functional needs repair` zeroes with the median of the other amount_tsh that are functional (REVIEW FOR OVERFITTING) - Rafa
- gps height lots of zeroes
- population lots of zeroes
- Map long/lat with region/city name
- public meeting NaN
- Make sure after shortlisting and hot encoding that all columns exist

## DevOps

- Set up AWS (Jupyter/Git integrations) - Went with Google Cloud

## Clear predactors suggestion - idea

in 2nd level of stacking add the columns that are CLEAR classifiers to the dataset


## Modeling

  - Emily
    - KNN
  - Rafa
    - XGBoost
  - Hatem
    - Multi-class regression
  - Theo
    - Random Forest
