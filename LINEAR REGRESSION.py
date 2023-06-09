import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
X = 2.5 * np.random.randn(5) + 1.5   
res = 0.5 * np.random.randn(5)      
y = 2 + 0.3 * X + res                 


df = pd.DataFrame(
    {'X': X,
     'y': y}
)


df
xmean = np.mean(X)
ymean = np.mean(y)


df['xycov'] = (df['X'] - xmean) * (df['y'] - ymean)
df['xvar'] = (df['X'] - xmean)**2


beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - (beta * xmean)
print(f'alpha = {alpha}')
print(f'beta = {beta}')
ypred = alpha + beta * X
print(ypred)
plt.figure(figsize=(12, 6))
plt.plot(X, ypred)     
plt.plot(X, y, 'rd')   
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')

plt.show()
