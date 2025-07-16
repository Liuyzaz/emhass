def format_timestamp(timestamp):
    """Format a timestamp to a more readable string."""
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def calculate_percentage(part, whole):
    """Calculate the percentage of a part relative to a whole."""
    if whole == 0:
        return 0
    return (part / whole) * 100

def load_json(file_path):
    """Load a JSON file and return its contents."""
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to a JSON file."""
    import json
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def flatten_list(nested_list):
    """Flatten a nested list into a single list."""
    return [item for sublist in nested_list for item in sublist]