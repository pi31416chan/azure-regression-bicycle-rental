from azure.ai.ml import load_component
from ml_base import COMPONENT_PATH, ml_client

# Load component from YAML file
linear_regression_component = load_component(
    source=COMPONENT_PATH / "linear_regression.yml"
)

ml_client.components.create_or_update(linear_regression_component)
