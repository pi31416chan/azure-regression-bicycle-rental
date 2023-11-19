from azure.ai.ml import load_component
from ml_base import COMPONENT_PATH, ml_client

# Load component from YAML file
split_train_test_component = load_component(
    source=COMPONENT_PATH / "split_train_test.yml"
)

ml_client.components.create_or_update(split_train_test_component)
