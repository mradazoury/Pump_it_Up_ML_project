# ML Proj Tasks

## Data prep

### DONE

- contstruction year => binning
- scheme name
  - irrelevant - elimanated

### Not tackled yet

- region and region code
  - doesn't have the same amount of categories
- amount_tsh
  - Lots os zeroes - doesn't make sense because if it's functional it can't have 0 "amount of water available to the water point"
  - Impute the `functional/functional needs repair` zeroes with the median of the other amount_tsh that are functional (REVIEW FOR OVERFITTING)
- water quality vs quality group
  - graph together to see patterns
- water point type / water point type group
- extraction type / extraction type group / extraction class


## DevOps

- Set up AWS (Jupyter/Git integrations)
