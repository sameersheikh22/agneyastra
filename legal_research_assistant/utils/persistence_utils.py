import json
import os
import datetime
import re

def save_results(filename_prefix: str, data_to_save: dict):
    """
    Saves the given data to a JSON file in the 'data/' directory.
    The filename will be <filename_prefix>_<timestamp>.json.

    Args:
        filename_prefix (str): A prefix for the filename.
        data_to_save (dict): The dictionary to save as JSON.
    """
    try:
        # Ensure the 'data/' directory exists in the project root
        # Assuming the script is run from the project root or `legal_research_assistant` parent.
        # More robust path handling might be needed if run from utils/ directly.
        data_dir = os.path.join(os.getcwd(), "legal_research_assistant", "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize filename_prefix to remove characters that are problematic for filenames
        sanitized_prefix = re.sub(r'[^a-zA-Z0-9_-]', '', filename_prefix)
        if not sanitized_prefix: # Ensure prefix is not empty after sanitization
            sanitized_prefix = "results"
            
        filename = f"{sanitized_prefix}_{timestamp}.json"
        filepath = os.path.join(data_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        
        print(f"Results successfully saved to: {filepath}")
        return filepath

    except IOError as e:
        # TODO: Replace print with logging
        print(f"Error saving results (IOError): {e}")
    except Exception as e:
        # TODO: Replace print with logging
        print(f"An unexpected error occurred while saving results: {e}")
    return None


def load_results(filename: str):
    """
    Loads data from a JSON file in the 'data/' directory.

    Args:
        filename (str): The name of the file (not the full path) to load from 'data/'.

    Returns:
        dict: The loaded data, or None if an error occurs.
    """
    try:
        # Assuming the 'data/' directory is relative to where this script might be called from
        # or relative to the project root.
        data_dir = os.path.join(os.getcwd(), "legal_research_assistant", "data")
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            # TODO: Replace print with logging
            print(f"Error: File not found at {filepath}")
            raise FileNotFoundError(f"No such file: '{filepath}'")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Results successfully loaded from: {filepath}")
        return data

    except FileNotFoundError:
        # Already handled above, but good to catch specifically if path logic changes
        # TODO: Replace print with logging
        print(f"Error: Could not find the file specified: {filename}")
    except json.JSONDecodeError as e:
        # TODO: Replace print with logging
        print(f"Error decoding JSON from file {filename}: {e}")
    except Exception as e:
        # TODO: Replace print with logging
        print(f"An unexpected error occurred while loading results: {e}")
    return None
