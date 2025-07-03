import yaml

def get_config(filename: str = "config.yaml") -> dict:
    """Loads the configuration from a YAML file."""
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)

def validate_config(config: dict) -> None:
    """Validate configuration structure and required fields."""
    required_fields = {
        "bot_token": str,
        "providers": dict,
        "models": dict,
        "permissions": dict
    }
    for field, expected_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
        if not isinstance(config[field], expected_type):
            raise TypeError(f"Config field {field} must be {expected_type.__name__}")
    if not config["models"]:
        raise ValueError("At least one model must be configured") 