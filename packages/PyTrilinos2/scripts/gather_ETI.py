import glob
import os
import sys

def get_list_of_ETI_files_to_include(list_all_ETI_files, list_all_classes_to_ETI):
    list_ETI_files = []
    for ETI_file in list_all_ETI_files:
        for ETI_class in list_all_classes_to_ETI:
            if ETI_file.startswith(ETI_class):
                list_ETI_files.append(ETI_file)
                break
    return list_ETI_files


def write_ETI_include_file(source_dir, filename, list_ETI_files):
    with open(source_dir+'/'+filename, 'w') as fh:
        fh.write('#ifndef PYTRILINOS2_TPETRA_ETI\n')
        fh.write('#define PYTRILINOS2_TPETRA_ETI\n\n')

        for ETI_file in list_ETI_files:
            fh.write('#include <'+ETI_file+'>\n')

        fh.write('\n#endif // PYTRILINOS2_TPETRA_ETI\n')
        

def write_ETI_getTpetraTypeName_file(source_dir, filename, list_ETI_files):
    with open(source_dir+'/'+filename, 'w') as fh:
        fh.write('from PyTrilinos2 import Tpetra\n\n')
        fh.write('def getTypeName(class_name, scalar_type, local_ordinal_type, global_ordinal_type, node_type):\n')

        for ETI_file in list_ETI_files:
            tmp = ETI_file[:ETI_file.index('.')].split("_")

            if tmp[0].lower() == 'tpetra':

                class_name = tmp[1].lower()
                class_name_internal = class_name
                if class_name == 'vector':
                    class_name_internal = 'Vector'
                if class_name == 'multivector':
                    class_name_internal = 'MultiVector'
                if class_name == 'crsgraph':
                    class_name_internal = 'CrsGraph'
                if class_name == 'crsmatrix':
                    class_name_internal = 'CrsMatrix'
                scalar_type = tmp[2].lower()
                scalar_type_internal = scalar_type
                if scalar_type == 'long':
                    scalar_type = 'long long'
                    scalar_type_internal = 'long_long'

                node_type = tmp[-1].lower()
                global_ordinal_type = tmp[-2].lower()
                global_ordinal_type_internal = global_ordinal_type
                if global_ordinal_type == 'long':
                    global_ordinal_type = 'long long'
                    global_ordinal_type_internal = 'long_long'
                    local_ordinal_type = tmp[-4].lower()
                else:
                    local_ordinal_type = tmp[-3].lower()
                local_ordinal_type_internal = local_ordinal_type
                if local_ordinal_type == 'long':
                    local_ordinal_type = 'long long'
                    local_ordinal_type_internal = 'long_long'

                if node_type == 'serial':
                    node_type_internal = 'Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace'
                if node_type == 'threads':
                    node_type_internal = 'Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Threads_Kokkos_HostSpace_t'
                if node_type == 'openmp':
                    node_type_internal = 'Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_OpenMP_Kokkos_HostSpace_t'
                if node_type == 'cuda':
                    node_type_internal = 'Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Cuda'
                fh.write('\tif class_name.lower() == "'+class_name+'" and scalar_type.lower() == "'+scalar_type+'" and local_ordinal_type.lower() == "'+local_ordinal_type+'" and global_ordinal_type.lower() == "'+global_ordinal_type+'" and node_type.lower() == "'+node_type+'":\n')
                fh.write('\t\treturn Tpetra.'+class_name_internal+'_'+scalar_type_internal+'_'+local_ordinal_type_internal+'_'+global_ordinal_type_internal+'_'+node_type_internal+'_t\n')
        fh.write('\tprint("Warning: Unknown type, the function returns None.")\n')
        fh.write('\treturn None\n')


if __name__ == '__main__':
    CMAKE_CURRENT_SOURCE_DIR = sys.argv[1]
    list_all_ETI_files = sys.argv[2]
    list_all_classes_to_ETI = sys.argv[3]
    output_file = sys.argv[4]

    with open(list_all_ETI_files, 'r') as fh:
        all_ETI_files = fh.read().splitlines()

    with open(list_all_classes_to_ETI, 'r') as fh:
        all_ETI_classes = fh.read().splitlines()
    
    reduce_list = True

    if reduce_list:
        print('all_ETI_files = '+str(all_ETI_files))
        print('all_ETI_classes = '+str(all_ETI_classes))
        list_ETI_files = get_list_of_ETI_files_to_include(all_ETI_files, all_ETI_classes)
        print('list_ETI_files = '+str(list_ETI_files))
    else:
        list_ETI_files = all_ETI_files
    write_ETI_include_file(CMAKE_CURRENT_SOURCE_DIR, output_file, list_ETI_files)
    write_ETI_getTpetraTypeName_file(CMAKE_CURRENT_SOURCE_DIR+'/python',  'getTpetraTypeName.py', list_ETI_files)
