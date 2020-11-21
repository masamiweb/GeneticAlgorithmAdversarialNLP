import os
import requests 
import tarfile
import shutil
import zipfile

   
def download(url: str, dest_folder: str):      
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("Downloading file ", filename, ", please wait ...")
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024): # set chunk size of 1MB (can reduce or increase depending on memory available) 
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        print("Download complete!")
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text)) 
    return


def extract_tar_file(file_path: str, destination_folder: str): 
    if not os.path.exists(file_path):
        print("tar file ",file_path, " not found!")
    else:
        print("\nExtracting file, please wait...")
        my_tar = tarfile.open(file_path)
        my_tar.extractall(destination_folder) # specify which folder to extract to
        my_tar.close()
        os.remove(file_path) # after extracting the contents, delete the tar file
        print("File extracted in folder: ",destination_folder)
    return
       

        
def get_directory_name(path):  
    dir_list = []
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            if file[0] != '.':
                dir_list.append(file)
    return dir_list


def extract_zip_file(file_path: str, destination_folder: str):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    os.remove(file_path)

    
def rename_folder(from_folder: str, to_folder: str):  
    print("Renaming folder: ", from_folder, " to: ", to_folder)
    os.rename(from_folder, to_folder)
    return

def rename_s140_file(folder_path):
    for entry in os.listdir(folder_path):
        if os.path.isdir(entry):
            continue
        values = entry.strip().split('.')
        if values[-1] == '.zip':
            os.remove(os.path.join(folder_path, entry))
        else:
            if entry.startswith('train'):
                os.rename(os.path.join(folder_path, entry),os.path.join(folder_path,'train.csv'))
                print("Renamed:\t", os.path.join(folder_path, entry), " To:\t", os.path.join(folder_path,'train.csv'))
            else:
                os.remove(os.path.join(folder_path, entry))
    
            
    