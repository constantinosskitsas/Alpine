import os
def get_max_previous_id(experimental_folder):
    folder_names = os.listdir(experimental_folder)
    max_id = 0
    for folder_name in folder_names:
        try:
            folder_id = int(folder_name.split('_')[-1])
            max_id = max(max_id, folder_id)
        except ValueError:
            continue
    return max_id

def generate_new_id(previous_ids):
    if isinstance(previous_ids, int):
        return previous_ids + 1
    elif isinstance(previous_ids, list) or isinstance(previous_ids, tuple):
        if previous_ids:
            return max(previous_ids) + 1
        else:
            return 1
    else:
        raise TypeError("previous_ids must be an integer or an iterable")
def create_new_folder(experimental_folder, new_id):
    new_folder_path = os.path.join(experimental_folder, f"_{new_id}")
    os.makedirs(new_folder_path)
    print(f"Created new folder: {new_folder_path}")
    
experimental_folder = "./data3_/res/"
new_id = generate_new_id(get_max_previous_id(experimental_folder))
print(new_id)
create_new_folder(experimental_folder, new_id)