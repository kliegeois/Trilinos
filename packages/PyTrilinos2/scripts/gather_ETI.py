import glob
import os
import sys

def get_list_of_ETI_files_to_include(list_all_ETI_files, list_all_classes_to_ETI):
    list_ETI_files = []
    for ETI_file in list_all_ETI_files:
        for ETI_class in list_all_classes_to_ETI:
            if ETI_file.startswith(ETI_class) and not ETI_file.startswith(ETI_class+'_DOUBLE_DOUBLE'):
                list_ETI_files.append(ETI_file)
                break
    return list_ETI_files


def write_ETI_file(source_dir, filename, list_ETI_files):
    with open(source_dir+'/'+filename, 'w') as fh:
        fh.write('#ifndef PYTRILINOS2_TPETRA_ETI\n')
        fh.write('#define PYTRILINOS2_TPETRA_ETI\n\n')

        for ETI_file in list_ETI_files:
            fh.write('#include <'+ETI_file+'>\n')

        fh.write('\n#endif // PYTRILINOS2_TPETRA_ETI\n')
        

if __name__ == '__main__':
    CMAKE_CURRENT_SOURCE_DIR = sys.argv[1]
    list_all_ETI_files = sys.argv[2]
    list_all_classes_to_ETI = sys.argv[3]
    output_file = sys.argv[4]

    with open(list_all_ETI_files, 'r') as fh:
        all_ETI_files = fh.read().splitlines()

    with open(list_all_classes_to_ETI, 'r') as fh:
        all_ETI_classes = fh.read().splitlines()

    print('all_ETI_files = '+str(all_ETI_files))
    print('all_ETI_classes = '+str(all_ETI_classes))
    list_ETI_files = get_list_of_ETI_files_to_include(all_ETI_files, all_ETI_classes)
    print('list_ETI_files = '+str(list_ETI_files))
    write_ETI_file(CMAKE_CURRENT_SOURCE_DIR, output_file, list_ETI_files)
