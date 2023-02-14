import anyscale
import ray
from ray.air.config import ScalingConfig
from ray.data.preprocessors import MinMaxScaler
from ray.train.xgboost import XGBoostTrainer

ray.init()

########
# DATA #
########

# Read Parquet file to Ray Dataset.
dataset = ray.data.read_parquet(
    "s3://anyscale-training-data/intro-to-ray-air/nyc_taxi_2021.parquet"
)

# Split data into training and validation subsets.
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

# Split datasets into blocks for parallel preprocessing.
# `num_blocks` should be lower than number of cores in the cluster.
train_dataset = train_dataset.repartition(num_blocks=5)
valid_dataset = valid_dataset.repartition(num_blocks=5)

# Define a preprocessor to normalize the columns by their range.
preprocessor = MinMaxScaler(columns=["trip_distance", "trip_duration"])

############
# TRAINING #
############

trainer = XGBoostTrainer(
    label_column="is_big_tip",
    num_boost_round=50,
    scaling_config=ScalingConfig(
        num_workers=5,
        use_gpu=False,  # True for the GPU training, 1 GPU per worker
    ),
    params={
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "tree_method": "approx",  # "gpu_hist" for GPU training
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
    preprocessor=preprocessor,
)

# Invoke training.
# The resulting object grants access to metrics, checkpoints, and errors.
result = trainer.fit()

training_accuracy = 1 - result.metrics['train-error']
validation_accuracy = 1 - result.metrics['valid-error']
iterations = result.metrics['training_iteration']

##############################
# Report metrics to Anyscale #
##############################

anyscale.job.output({
    "training_iterations": iterations,
    "training_accuracy": training_accuracy,
    "validation_accuracy": validation_accuracy
})

# Further reading
# 1. Tutorial on Batch Training with Ray Datasets: https://docs.ray.io/en/master/data/examples/batch_training.html
# 2. Ray Data - key concepts: https://docs.ray.io/en/master/data/key-concepts.html
