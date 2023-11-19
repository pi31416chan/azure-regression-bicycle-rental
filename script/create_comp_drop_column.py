from azure.ai.ml import load_component
from ml_base import COMPONENT_PATH, ml_client

# Load component from YAML file
drop_column_component = load_component(
    source=COMPONENT_PATH / "drop_column.yml"
)

ml_client.components.create_or_update(drop_column_component)
