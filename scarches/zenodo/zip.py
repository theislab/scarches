def zip_model_directory(output_path: str = "./unknown", directory: str = "./"):
    import shutil
    try:
        shutil.make_archive(f"{output_path}", 'zip', directory)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {directory}")


def unzip_model_directory(file_name: str = "unknown", extract_dir: str = "./"):
    import shutil
    try:
        shutil.unpack_archive(file_name, extract_dir, format='zip')
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {file_name}")
