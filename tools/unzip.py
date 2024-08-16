import tarfile
from glob import glob
import os
from tqdm import tqdm
def un_tar(file_name):\
    file = os.path.basename(file_name)
    save_file = os.path.splitext(file)[0]
    tar = tarfile.open(file_name)
    names = tar.getnames()
    for name in tqdm(names):
        new_save = "/".join(os.path.dirname(file_name).split("/")[:-1])
        tar.extract(name, new_save + "/ImageNet_train/" + save_file)
        tar.close()
if __name__ == "__main__":
    floder = "/home/guest/YLxiaFiles/ImageNet/ILSVRC2012_img_train/*.tar"
    record = open("record.txt", "w")
    for i in glob(floder):
        try:un_tar(i)
        except:record.write(str(i) + "\n")
        continue