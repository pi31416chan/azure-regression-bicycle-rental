from azure.ai.ml import load_component
from ml_base import COMPONENT_PATH, ml_client

# Load component from YAML file
split_x_y_component = load_component(source=COMPONENT_PATH / "split_x_y.yml")

ml_client.components.create_or_update(split_x_y_component)
