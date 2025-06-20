import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import text_format
from tensorflow_model_remediation import min_diff
acs_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/acsincome_raw_2018.csv")
acs_df.sample(5)
LABEL_KEY = 'PINCP'
LABEL_THRESHOLD = 50000.0
acs_df[LABEL_KEY] = acs_df[LABEL_KEY].apply(
    lambda income: 1 if income > LABEL_THRESHOLD else 0)
acs_df.sample(10)
inputs = {}
features = acs_df.copy()
features.pop(LABEL_KEY)

# Instantiate a Keras input node for each column in the dataset.
for name, column in features.items():
  if name != LABEL_KEY:
    inputs[name] = tf.keras.Input(
        shape=(1,), name=name, dtype=tf.float64)

# Stack the inputs as a dictionary and preprocess them.
def stack_dict(inputs, fun=tf.stack):
  values = []
  for key in sorted(inputs.keys()):
    values.append(tf.cast(inputs[key], tf.float64))

  return fun(values, axis=-1)

x = stack_dict(inputs, fun=tf.concat)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(features)))

x = normalizer(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

base_model = tf.keras.Model(inputs, outputs)

METRICS = [
  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
  tf.keras.metrics.AUC(name='auc'),
]

base_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=METRICS)

def dataframe_to_dataset(dataframe):
  dataframe = dataframe.copy()
  labels = dataframe.pop(LABEL_KEY)
  dataset = tf.data.Dataset.from_tensor_slices(
      ((dict(dataframe), labels)))
  return dataset

RANDOM_STATE = 200
BATCH_SIZE = 100
EPOCHS = 10

acs_train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE)
acs_train_ds = dataframe_to_dataset(acs_train_df)
acs_train_batches = acs_train_ds.batch(BATCH_SIZE)

base_model.fit(acs_train_batches, epochs=EPOCHS)

acs_test_df = acs_df.drop(acs_train_df.index).sample(frac=1.0)
acs_test_ds = dataframe_to_dataset(acs_test_df)
acs_test_batches = acs_test_ds.batch(BATCH_SIZE)

base_model.evaluate(acs_test_batches, batch_size=BATCH_SIZE)

base_model_predictions = base_model.predict(
    acs_test_batches, batch_size=BATCH_SIZE)

SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}
SENSITIVE_ATTRIBUTE_KEY = 'SEX'
PREDICTION_KEY = 'PRED'

base_model_analysis = acs_test_df.copy()
base_model_analysis[SENSITIVE_ATTRIBUTE_KEY].replace(
    SENSITIVE_ATTRIBUTE_VALUES, inplace=True)
base_model_analysis[PREDICTION_KEY] = base_model_predictions

base_model_analysis.sample(5)

eval_config_pbtxt = """
  model_specs {
    prediction_key: "%s"
    label_key: "%s" }
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics { class_name: "BinaryAccuracy" }
    metrics { class_name: "AUC" }
    metrics { class_name: "ConfusionMatrixPlot" }
    metrics {
      class_name: "FairnessIndicators"
      config: '{"thresholds": [0.50]}'
    }
  }
  slicing_specs {
    feature_keys: "%s"
  }
  slicing_specs {}
""" % (PREDICTION_KEY, LABEL_KEY, SENSITIVE_ATTRIBUTE_KEY)
eval_config = text_format.Parse(eval_config_pbtxt, tfma.EvalConfig())

base_model_eval_result = tfma.analyze_raw_data(base_model_analysis, eval_config)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    base_model_eval_result)

MIN_DIFF_BATCH_SIZE = 50
sensitive_group_ds = dataframe_to_dataset(sensitive_group_pos)
non_sensitive_group_ds = dataframe_to_dataset(non_sensitive_group_pos)

sensitive_group_batches = sensitive_group_ds.batch(
    MIN_DIFF_BATCH_SIZE, drop_remainder=True)
non_sensitive_group_batches = non_sensitive_group_ds.batch(
    MIN_DIFF_BATCH_SIZE, drop_remainder=True)
min_diff_model = min_diff.keras.MinDiffModel(
    original_model=base_model,
    loss=min_diff.losses.MMDLoss(),
    loss_weight=1)

min_diff_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=METRICS)
min_diff_model.fit(acs_train_min_diff_ds, epochs=EPOCHS)
min_diff_model.evaluate(acs_test_batches, batch_size=BATCH_SIZE)
min_diff_model_predictions = min_diff_model.predict(
    acs_test_batches, batch_size=BATCH_SIZE)
min_diff_model_analysis = acs_test_df.copy()
min_diff_model_analysis[SENSITIVE_ATTRIBUTE_KEY].replace(
    SENSITIVE_ATTRIBUTE_VALUES, inplace=True)
min_diff_model_analysis[PREDICTION_KEY] = min_diff_model_predictions
min_diff_model_eval_result = tfma.analyze_raw_data(
    min_diff_model_analysis, eval_config)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    min_diff_model_eval_result)
