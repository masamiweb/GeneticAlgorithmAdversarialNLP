from manny_modules import download_utils as dutils

def download_dataset():
    from_folder =""
    to_folder = DATA_SET_LOCATION 

    if DATASET == 1:
        dutils.download(DATASET_URL, dest_folder=DATASET_FOLDER)
        dutils.extract_tar_file(TAR_FILE_PATH, DATASET_FOLDER)
        directories_in_dataset_folder = dutils.get_directory_name(DATASET_FOLDER)

        if len(directories_in_dataset_folder) == 1:
            from_folder=DATASET_FOLDER+"/"+directories_in_dataset_folder[0]
            dutils.rename_folder(from_folder, to_folder)
            print("Done!")
        else:
            print("ERROR!")
            print(DATASET_FOLDER," folder has too many sub directories!")
            print("Make sure ", DATASET_FOLDER, " is empty before downloading fresh dataset file")
        
    if DATASET == 2:
        dutils.download(DATASET_URL, dest_folder=DATASET_FOLDER)
        dutils.extract_zip_file(TAR_FILE_PATH, DATASET_FOLDER)
        dutils.rename_s140_file(DATASET_FOLDER)
    