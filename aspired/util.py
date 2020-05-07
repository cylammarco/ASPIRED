import os


def check_files(paths):
    '''
    Go through the filelist provided and check if all files exist.
    '''

    for filepath in paths:
        try:
            os.path.isfile(filepath)
        except:
            ValueError('File ' + filepath + ' does not exist.')
