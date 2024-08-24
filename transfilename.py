import os

def rename_files_to_sequence(folder_path, extension='jpg'):
    """
    Rename all files in the given folder to a sequential order with the specified extension.

    Parameters:
    folder_path (str): The path to the folder containing the files.
    extension (str): The extension to use for the renamed files (default is 'jpg').

    Returns:
    None
    """
    if not os.path.isdir(folder_path):
        print(f"The provided path '{folder_path}' is not a directory.")
        return

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # Sort the files if needed

    for index, filename in enumerate(files, start=1):
        new_name = f"{index}.{extension}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed '{filename}' to '{new_name}'")

    print("Renaming completed.")

# Example usage:
path = r'D:\data\tiny\weather'
rename_files_to_sequence(path)