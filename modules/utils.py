def extract_timestamp(filepath, sep='_'):
    """ Extracts the timestamp from the filename in the format '<directory>/YYYYMMDDHHMMSS.path(.pt)'.
    Args:
        filepath (str): Path to the file from which to extract the timestamp.
        sep (str): Separator used in the filename. Default is '_'.
    Returns:
        str: The extracted timestamp in the format sep + 'YYYYMMDDHHMMSS', or '' if not found.
    Raises:
        ValueError: If the file path does not exist, is not a string, or does not end with '.path'or '.pt'.
        ValueError: If the filename does not match the expected format.
        ValueError: If the timestamp is not found in the filename.
    """
    import re, os
    if not os.path.exists(filepath):
        raise ValueError(f"File path '{filepath}' does not exist.")
    if not isinstance(filepath, str):
        raise ValueError("File path must be a string.")
    if not (filepath.endswith('.path') or filepath.endswith('.pt')):
        raise ValueError("File path must end with '.path' or .'pt'.")
    if not isinstance(sep, str):
        raise ValueError("Separator must be a string.")
    filename = os.path.basename(filepath)
    match = re.search(r'(\d{12})\.(path|pt)$', filename)
    if match:
        return sep + match.group(1)
    else:
        return ''
    
def configure_logging(logname, logfile, loglevel, timestamp=None):
    import logging, os
    logger = logging.getLogger(logname)
    # Remove all existing handlers
    logger.handlers.clear()
    match loglevel:
        case 'notset':
            logger.setLevel(logging.NOTSET)
        case 'debug':
            logger.setLevel(logging.DEBUG)
        case 'info':
            logger.setLevel(logging.INFO)
        case 'warning':
            logger.setLevel(logging.WARNING)
        case 'error':
            logger.setLevel(logging.ERROR)
        case 'critical':
            logger.setLevel(logging.CRITICAL)
    if logfile:
        if timestamp:
            logger.info(f'{timestamp=}')
            logfile = logfile.replace('.log', f'_{timestamp}.log')
        os.makedirs(os.path.dirname(logfile), exist_ok=True)  # Ensure directory exists
        file_handler = logging.FileHandler(logfile, 'w+')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger