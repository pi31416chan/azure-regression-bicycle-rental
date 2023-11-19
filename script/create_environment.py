from datetime import datetime

from azure.ai.ml.entities import Environment
from ml_base import ml_client

basic_env = Environment(
    name="basic",
    image="docker.io/pi31416chan/az-basic:latest",
    description="Basic environment for lightweight task like data processing",
    tags={
        "project": "regression-bicycle",
        "python": "3.10",
        "dependencies": "azure-ai-ml mltable pandas numpy",
        "created at": datetime.now().strftime("%Y-%m-%d-%H%M"),
    },
)
ml_client.environments.create_or_update(basic_env)

sklearn_env = Environment(
    name="sklearn",
    image="docker.io/pi31416chan/az-sklearn:latest",
    description="Environment for machine learning with scikit-learn",
    tags={
        "project": "regression-bicycle",
        "python": "3.10",
        "dependencies": "azure-ai-ml mltable pandas numpy sklearn mlflow",
        "created at": datetime.now().strftime("%Y-%m-%d-%H%M"),
    },
)
ml_client.environments.create_or_update(sklearn_env)
