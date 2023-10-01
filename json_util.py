import json

def load_category_names(json_file):
    """
    Load category names from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        category_names (dict): A dictionary mapping category numbers to category names.
    """
    with open(json_file, 'r') as f:
        category_names = json.load(f, strict=False)
    # print("Inside load_category_names:", category_names) 
    return category_names