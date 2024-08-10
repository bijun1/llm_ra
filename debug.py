import subprocess
import pickle
from typing import Optional, Tuple
from datasets import load_dataset
from llmRA.utils import *
import multiprocessing
_DEFAULT_CMD_TIMEOUT = 20
def _run_command(command: str, stdin: Optional[str] = None, timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT) -> Tuple[str, str]:
    output = subprocess.run(command.split(), capture_output=True, text=True, input=stdin, timeout=timeout)
    stdout = output.stdout.decode('utf-8') if isinstance(output.stdout, bytes) else output.stdout
    stderr = output.stderr.decode('utf-8') if isinstance(output.stderr, bytes) else output.stderr
    return stdout, stderr

def formCode(row):
    code = ""
    if row["real_deps"] is not None:
        code += row["real_deps"]
    elif row["synth_deps"] is not None:
        code += row["synth_deps"]
    code += row["func_def"]
    return code

mac_compile_postfix = " -target x86_64-pc-linux-gnu -I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include "

def extract(name, folder_path):
    res = {}
    res["name"] = name
    for filename in os.listdir(folder_path):
        if filename.endswith("_intervals"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                res["interval"] = content
        elif filename.endswith("_edges"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                res["edges"] = content
    return res

def work(ids):
    dataset = load_dataset('jordiae/exebench', split='train_synth_compilable') # , use_auth_token=True)
    for i in ids:
        print("Running ", i)
        row = dataset[i]
        code = formCode(row)
        folder_path = "./.Programs/Prog_%d/"%i
        create_folder(folder_path)
        fpath = folder_path + "T_%d_func.c"%i
        with open(fpath, "w", encoding="utf-8") as file:
            file.write(code)
        cmd = "clang -O3 -S " + fpath + " -mllvm --regalloc=grad -mllvm -dump-info -mllvm -ra-path-prefix=%s"%folder_path
        cmd += " -mllvm -ra-score-log=%s.%d_log "%(folder_path, i) + mac_compile_postfix
        cmd += " -o t.s"
        try:
            out, err = _run_command(cmd)
        except Exception as e:
            print(cmd)
            print(e)
            continue
        print(err)

if __name__ == '__main__':
    ids = [
            13477,
            23378,
            23754,
            24061,
            4660,
            14589,
            27082
            ]
    create_folder(".results")
    create_folder(".errors")
    work(ids)
    
