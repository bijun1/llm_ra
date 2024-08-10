import os

def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(e)
        assert 0

def read_project_file(project_name):
    filename = ".%s" % project_name
    data = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                name, value, mcnt, lcnt, scnt, lscnt, ercnt, crcnt = line.strip().split(',')
                data[name] = [float(value), float(mcnt), float(lcnt), float(scnt), float(lscnt), float(ercnt), float(crcnt)]
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
    except ValueError:
        print(f"Error processing line: {line.strip()}")
    return data

def remove_project_file(project_name):
    file_path = f".{project_name}"
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")
    else:
        print(f"File {file_path} does not exist.")

def write_list_to_file(filename, my_list):
    try:
        with open(filename, 'w') as file:
            for item in my_list:
                file.write(f"{item}\n")
    except Exception as e:
        print(f"An error occurred: {e}")

def remove_additional_files(fnames):
    posts = ["intervals", "edges"]
    for fname in fnames:
        for post in posts:
            path = ".%s_%s"%(fname, post)
            os.remove(path)
