# ML Proj Tasks

## Process

 - Split Notebooks and take out repeated functions to seperate `.py` files - 70% done

## Data prep

### DONE

- contstruction year => binning
- scheme name
  - irrelevant - elimanated

### Not tackled yet (Feature Engineering)

- region and region code
  - doesn't have the same amount of categories
- amount_tsh
  - Lots os zeroes - doesn't make sense because if it's functional it can't have 0 "amount of water available to the water point"
  - Impute the `functional/functional needs repair` zeroes with the median of the other amount_tsh that are functional (REVIEW FOR OVERFITTING) - Rafa
- water quality vs quality group
  - graph together to see patterns
- water point type / water point type group
- extraction type / extraction type group / extraction class
- Map long/lat with region/city name

## DevOps

- Set up AWS (Jupyter/Git integrations)

## Modeling

  - Emily
    - KNN
  - Rafa
    - XGBoost
  - Hatem
    - Multi-class regression
  - Theo
    - Random Forest
