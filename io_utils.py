from datetime import date
from pathlib import Path


def generate_path_in_output_dir(config, file, add_version=False, add_date_timestamp=False):
    file_path = Path(config.outputs.output_dir, file)
    if add_version:
        file_path = file_path.with_stem(file_path.stem + f"_data_v_{config.data_loading.version}")
    if add_date_timestamp:
        file_path = file_path.with_stem(file_path.stem + f"_ts_{date.today().isoformat()}")

    return file_path
