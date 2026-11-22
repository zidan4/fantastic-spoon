import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris['feature_names'])
print(df.head(5))

np.random.seed(42)
X = df.values
y = iris.target

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )
