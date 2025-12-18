import logging 
import os
from datetime import datetime
import json


class LogWriter:
    """
    Logger object class for logging messages to a file and console
    """

    def __init__(self, app_name: str, 
                       log_root: str):
        """
        Initialize the logger

        Args:
        -----
        app_name : str
            Name of the application for which logs are being created
        log_dir : str
            Directory where log files will be stored (default: 'logs')
        """

        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)


        handlers = {'console_handler': logging.StreamHandler()}

        self.log_root = os.path.join(log_root, app_name)

        if self.log_root:

            os.makedirs(os.path.join(log_root, 'logs'), exist_ok=True)  

            date_str = datetime.now().strftime('%Y-%m-%d')
            filename = f'{app_name}_{date_str}.log'

            handlers['file_handler'] = logging.FileHandler(os.path.join(self.log_root, 'logs', f'{filename}.log'), mode='w')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in handlers.values():
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)



    def log_metrics(self, metrics: dict, 
                          iteration: int):
        """
        Log metrics to the console and file for a specific epoch

        Args:
        -----
        metrics : dict
            Dictionary containing metric names and their values
        iteration : int
            Current iteration number    
        """

        metrics_string = f'[Iteration {iteration}] ' + ', '.join(f'{key}: {value}' for key, value in metrics.items())
        self.logger.info(metrics_string)

    def log_custom_message(self, message: str):
        """
        Log a custom message to the console and file

        Args:
        -----
        message : str
            Custom message to log
        """
        self.logger.info(message)


    def log_dict_metrics(self, metrics: dict, 
                               metric_name: str):
        """
        Log weighted metric to console and save to JSON file for each class

        Args:
        -----
        metrics : dict
            Target metric values for each class 

        metric_name : str
            Name of calculated metric
        """

        self.log_custom_message(f'{metric_name}: {metrics["average_score"]}')

        json_file_path = os.path.join(self.log_root, f'{metric_name}.json')

        with open(json_file_path, 'w') as f:    
            json.dump(metrics, f)