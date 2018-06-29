# move one file from one place to the other
import shutil
import glob
import os

def move_one_file(source_path, target_path):
    if not os.path.exists(source_path):
        print ("%s not exists" %(source_path))
    else:
         shutil.copyfile(source_path, target_path)

#move all file in the source dir to the target dir 
def move_all_file(source_dir_path, target_dir_path):
    if not os.path.exists(source_dir_path):
        print ("%s not exists" %(source_dir_path))
    else:
        for source_pic_path in glob.glob(source_dir_path+'/*'):
            basename = os.path.basename(source_pic_path)
            target_pic_path = os.path.join(target_dir_path, basename)
            move_one_file(source_pic_path, target_pic_path)
