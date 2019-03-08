# ML Proj Tasks

## Process

 - Split Notebooks and take out repeated functions to seperate `.py` files - 70% done

## Data prep

### DONE

- contstruction year => binning
- scheme name
  - irrelevant - elimanated

### Not tackled yet (Feature Engineering)

- * region and region code - Do Nothing
- amount_tsh
  - Lots os zeroes - doesn't make sense because if it's functional it can't have 0 "amount of water available to the water point"
  - Impute the `functional/functional needs repair` zeroes with the median of the other amount_tsh that are functional (REVIEW FOR OVERFITTING) - Rafa
- gps height lots of zeroes
- population lots of zeroes
- * water quality vs quality group - Do Nothing
- * water point type / water point type group - Drop waterpoint_type_group
- * extraction type / extraction type group / extraction class - Do Nothing
- Map long/lat with region/city name
- public meeting NaN
- Make sure after shortlisting and hot encoding that all columns exist

## DevOps

- Set up AWS (Jupyter/Git integrations) - Went with Google Cloud

## Modeling

  - Emily
    - KNN
  - Rafa
    - XGBoost
  - Hatem
    - Multi-class regression
  - Theo
    - Random Forest
