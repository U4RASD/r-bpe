import logging
import os

def setup_logger(app_name='BPE'):
    log_dir = 'logs'
    log_file_name = f'{app_name}.log'
    log_file_path = os.path.join(log_dir, log_file_name)
    warn_error_log_file_name = f'{app_name}_warn_error.log' 
    warn_error_log_file_path = os.path.join(log_dir, warn_error_log_file_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(app_name)
    
    # check if logger already has handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    warn_error_file_handler = logging.FileHandler(warn_error_log_file_path, mode='a', encoding='utf-8')
    warn_error_file_handler.setLevel(logging.WARNING)  

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    warn_error_file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(warn_error_file_handler)  

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
