# Import
from configparser import ConfigParser
from os.path import dirname
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Getting file path
SCRIPT_PATH = Path(dirname(__file__))
PROJECT_PATH = SCRIPT_PATH.parent
COMPONENT_PATH = SCRIPT_PATH / "component"

# Parsing config
config = ConfigParser()
config.read(f"{SCRIPT_PATH}/config.ini")
base_config = config["base_config"]
credential = config["credential"]


# Creating ML client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=credential["subscription_id"],
    resource_group_name=base_config["resource_group"],
    workspace_name=base_config["workspace"],
)
