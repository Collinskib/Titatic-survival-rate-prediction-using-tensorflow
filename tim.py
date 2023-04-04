import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

raw=pd.read_csv('train.csv')
dftrain = pd.read_csv('train.csv')
dfeval = pd.read_csv('eval.csv')
ytrain = dftrain.pop('survived')
yeval = dfeval.pop('survived')
print(ytrain.shape)
dftrain.sex.value_counts().plot(kind='barh')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERICAL_COLUMNS = ['age', 'fare']
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
print(feature_columns)


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


train_input_fn = make_input_fn(dftrain, ytrain)
eval_input_fn = make_input_fn(dfeval, yeval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[45])
print(yeval.loc[45])
print(result[45]['probabilities'][1])

predict_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(raw, y=None, shuffle=False)

predictions = list(linear_est.predict(predict_input_fn))
probs = [f"{p['probabilities'][1]*100:.2f}%" for p in predictions]

prob1 = probs

raw['Survival rate'] = prob1

# Save the updated data frame to a new CSV file
raw.to_csv('train_with_predictions.csv', index=False)

