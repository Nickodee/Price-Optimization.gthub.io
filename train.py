import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

df = pd.read_csv("D:\Price Optimization/amazon.csv")

df['actual_price'] = df['actual_price'].str.replace('₹', '') # remove currency symbol
df['actual_price'] = df['actual_price'].str.replace(',', '') # remove commas
df['actual_price'] = df['actual_price'].astype(float) # convert to float

df['discount_percentage'] = df['discount_percentage'].astype(str)
df['discount_percentage'] = df['discount_percentage'].str.replace('%', '')
df['discount_percentage'] = df['discount_percentage'].astype(float)

y = df.discounted_price

features = ['actual_price', 'discount_percentage', 'rating']

X = df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

train_y = train_y.str.replace('₹', '')
train_y = train_y.str.replace(',', '')
train_y = train_y.astype(float)

rf_model = RandomForestRegressor(random_state= 0)

rf_model.fit(train_X, train_y)

val_X = val_X.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
val_X = val_X.dropna()
pickle.dump(rf_model, open('rf_model.pk1','wb'))
rf_model = pickle.load(open('rf_model.pk1', 'rb'))
print(rf_model.predict([[999, 50, 4.2]]))