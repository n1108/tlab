import pandas as pd
s = pd.Series([0.0, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.0, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01])
print(s.nunique())
print(s.value_counts(normalize=True))
