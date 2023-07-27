import logging
from pathlib import Path
from typing import Optional

from time_utils import get_now_timestemp_as_string


def set_file_logger(logging_file_path: Path):
    cur_time_str = get_now_timestemp_as_string()
    new_file_name = f"{cur_time_str}_{logging_file_path.name}"
    logging_file_path = logging_file_path.with_name(new_file_name)
    logging.basicConfig(filename=logging_file_path, level=logging.INFO)
