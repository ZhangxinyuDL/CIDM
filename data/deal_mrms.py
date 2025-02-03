import h5py
from natsort import natsorted
import os
import pygrib
import gzip
import sys
import time

error = open("./error.txt", "a")
error.write("----------------------error----------------------" + "\n")


def deal():
    src = '/media/data8T/SHSR/2017/'
    gz = '/media/data8T/SHSR/gz_2017/'
    tar = '/media/data8T/SHSR/SHSR/2017/'

    files = os.listdir(src)
    files = natsorted(files)

    for i in range(0, len(files)):  
        try:
            # 解压
            g_file = gzip.GzipFile(src + files[i])
            # g_file = gzip.GzipFile(".grib2.gz")
            data = open(gz + files[i].replace(".gz", ""), "wb+")
            data.write(g_file.read())
            g_file.close()
            data.close()
        except Exception as e:
            # 解压失败
            print("解压失败   " + str(files[i]) + "     " + str(e))
            error.write("解压失败   " + str(files[i]) + "     " + str(e) + "\n")
            error.flush()
            continue
            # sys.exit()

        # pygrib open
        data = pygrib.open(gz + files[i].replace(".gz", ""))

        obj = tar + files[i][18:33] + ".hd5"
        f = h5py.File(obj, "w")  
        f.create_dataset("data", data=data[1].values, compression="gzip", compression_opts=4)
        f.close()
        data.close()

        # delete
        os.remove(gz + files[i].replace(".gz", ""))
        print(obj + "  -------  " + str(i))


deal()
error.close()
