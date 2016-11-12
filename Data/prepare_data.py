import os
from Util import list_filenames_from_dir, write_list_to, read_content_from_file, string_filter
from Parse import retrieve_from_xml

src_dir = 'E:\\data\\pan_dataset\\dataset'
save_dir = 'E:\\data\\pan_dataset\\docs'

if __name__ == '__main__':
    for full_name in list_filenames_from_dir(src_dir):
        filename = full_name.split('.')[0]
        affix = full_name.split('.')[1]
        if affix == 'xml':
            lines = [ string_filter(x) for x in retrieve_from_xml(read_content_from_file(os.path.join(src_dir, full_name)))]
            write_list_to(os.path.join(save_dir, filename + '.txt'), lines)



