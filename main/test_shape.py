import sys
from src.core.config import load_config
from src.data.data_loading import load_datasets
from src.data.preprocessing import preprocess_data

cfg = load_config(train_mode=True, path="configs/unet/tests/test_exp6.yaml")
datasets = load_datasets(cfg)
X, y_train, y_test = datasets[0], datasets[1], datasets[2]
x_train_tensor, _, _, _ = preprocess_data(cfg, X, y_train, y_test)
print("X_TRAIN_TENSOR SHAPE:", x_train_tensor.shape)
