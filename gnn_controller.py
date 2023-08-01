
import subprocess as sp
import time
import psutil
import os
import shutil

script = 'gnn.py'

if __name__ == "__main__":
    print('Running Script File:', script)
    time.sleep(5)
    run = sp.Popen(['python3', script])
    while True:
        time.sleep(5)
        print("PSUTIL:MEMORY:", psutil.virtual_memory().percent)
        if psutil.virtual_memory().percent >= 75:
            sp.Popen.terminate(run)
            run = sp.Popen(['python3', script])
