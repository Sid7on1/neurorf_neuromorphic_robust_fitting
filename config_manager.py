import yaml
import numpy as np
import logging
from typing import Dict, Any
from enum import Enum
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration management for SNN parameters and hardware settings.

    This class provides methods to load, validate, and generate configuration files.
    It also includes helper classes and utilities for configuration management.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize the ConfigManager instance.

        Args:
        - config_file (str): The path to the configuration file.
        """
        self.config_file = config_file
        self.config = None
        self.lock = Lock()

    def load_config(self, config_file: str = None) -> Dict[str, Any]:
        """
        Load the configuration from a YAML file.

        Args:
        - config_file (str): The path to the configuration file.

        Returns:
        - config (Dict[str, Any]): The loaded configuration.
        """
        if config_file is None:
            config_file = self.config_file

        if config_file is None:
            raise ValueError("Config file path is required")

        try:
            with open(config_file, 'r') as file:
                self.config = yaml.safe_load(file)
                return self.config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise

    def validate_parameters(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration parameters.

        Args:
        - config (Dict[str, Any]): The configuration to validate.

        Returns:
        - valid (bool): True if the configuration is valid, False otherwise.
        """
        required_keys = ['snn', 'hardware']
        if not all(key in config for key in required_keys):
            logger.error("Missing required configuration keys")
            return False

        snn_config = config['snn']
        hardware_config = config['hardware']

        # Validate SNN parameters
        if 'num_neurons' not in snn_config or not isinstance(snn_config['num_neurons'], int):
            logger.error("Invalid SNN num_neurons parameter")
            return False

        if 'learning_rate' not in snn_config or not isinstance(snn_config['learning_rate'], float):
            logger.error("Invalid SNN learning_rate parameter")
            return False

        # Validate hardware parameters
        if 'num_cores' not in hardware_config or not isinstance(hardware_config['num_cores'], int):
            logger.error("Invalid hardware num_cores parameter")
            return False

        if 'memory_size' not in hardware_config or not isinstance(hardware_config['memory_size'], int):
            logger.error("Invalid hardware memory_size parameter")
            return False

        return True

    def generate_default_config(self) -> Dict[str, Any]:
        """
        Generate a default configuration.

        Returns:
        - config (Dict[str, Any]): The default configuration.
        """
        config = {
            'snn': {
                'num_neurons': 100,
                'learning_rate': 0.01
            },
            'hardware': {
                'num_cores': 4,
                'memory_size': 1024
            }
        }
        return config

    def save_config(self, config: Dict[str, Any], config_file: str) -> None:
        """
        Save the configuration to a YAML file.

        Args:
        - config (Dict[str, Any]): The configuration to save.
        - config_file (str): The path to the configuration file.
        """
        with self.lock:
            try:
                with open(config_file, 'w') as file:
                    yaml.dump(config, file)
            except Exception as e:
                logger.error(f"Failed to save config file: {e}")
                raise

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
        - config (Dict[str, Any]): The current configuration.
        """
        if self.config is None:
            raise ValueError("Config not loaded")
        return self.config

class ConfigException(Exception):
    """
    Custom exception for configuration-related errors.
    """
    pass

class ConfigKey(Enum):
    """
    Enum for configuration keys.
    """
    SNN = 'snn'
    HARDWARE = 'hardware'
    NUM_NEURONS = 'num_neurons'
    LEARNING_RATE = 'learning_rate'
    NUM_CORES = 'num_cores'
    MEMORY_SIZE = 'memory_size'

def main():
    config_manager = ConfigManager()
    config = config_manager.generate_default_config()
    print(config)

    config_file = 'config.yaml'
    config_manager.save_config(config, config_file)

    loaded_config = config_manager.load_config(config_file)
    print(loaded_config)

    valid = config_manager.validate_parameters(loaded_config)
    print(f"Config valid: {valid}")

if __name__ == "__main__":
    main()