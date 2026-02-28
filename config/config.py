import os
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(var_name: str):
    value = os.getenv(var_name)

    if value is None:
        raise ValueError(f"Environment variable '{var_name}' is required but not set")

    return value

api_token = get_env_variable("API_TOKEN")
models_dir = get_env_variable("MODELS_DIR")
base_url = get_env_variable("BASE_URL")
odds_url = get_env_variable("ODDS_URL")