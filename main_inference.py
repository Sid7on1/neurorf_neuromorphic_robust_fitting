import argparse
import logging
import os
import sys
from typing import Dict, List

import config_manager
import cpu_simulation
import energy_profiler
import neuro_rf_snn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MainInference:
    def __init__(self):
        self.config = config_manager.ConfigManager()
        self.energy_profiler = energy_profiler.EnergyProfiler()

    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='NeuroRF Inference')
        parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
        parser.add_argument('--mode', type=str, default='lohi2', choices=['lohi2', 'cpu'], help='Execution mode')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        return parser.parse_args()

    def select_execution_mode(self, args: argparse.Namespace) -> None:
        if args.mode == 'lohi2':
            self.execute_on_lohi2(args)
        elif args.mode == 'cpu':
            self.execute_on_cpu(args)

    def execute_on_lohi2(self, args: argparse.Namespace) -> None:
        logger.info('Executing on Loihi 2...')
        # Load Loihi 2 configuration
        lohi2_config = self.config.load_lohi2_config(args.config)
        # Initialize Loihi 2 simulation
        lohi2_simulation = cpu_simulation.CPUSimulation(lohi2_config)
        # Run NeuroRF on Loihi 2
        neuro_rf_snn.NeuroRFSNN(lohi2_simulation).run()
        # Profile energy consumption
        self.energy_profiler.profile_energy_consumption(lohi2_simulation)

    def execute_on_cpu(self, args: argparse.Namespace) -> None:
        logger.info('Executing on CPU...')
        # Load CPU configuration
        cpu_config = self.config.load_cpu_config(args.config)
        # Initialize CPU simulation
        cpu_simulation = cpu_simulation.CPUSimulation(cpu_config)
        # Run NeuroRF on CPU
        neuro_rf_snn.NeuroRFSNN(cpu_simulation).run()
        # Profile energy consumption
        self.energy_profiler.profile_energy_consumption(cpu_simulation)

    def main(self) -> None:
        args = self.parse_arguments()
        self.select_execution_mode(args)
        logger.info('Inference completed successfully.')

if __name__ == '__main__':
    main_inference = MainInference()
    main_inference.main()