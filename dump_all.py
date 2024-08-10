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

def work(id, s_id, e_id):
    print("Processing %d to %d"%(s_id, e_id))
    dataset = load_dataset('jordiae/exebench', split='train_synth_compilable') # , use_auth_token=True)
    results = []
    for i in range(s_id, e_id):
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
    with open(".results/%i.pkl"%id, 'wb') as f :
        pickle.dump(results, f)

if __name__ == '__main__':
    processes = []
    assigns = [
        [0, 10000],
        [10000, 20000],
        [20000, 30000]
    ]
    create_folder(".results")
    create_folder(".errors")
    for i, assign in enumerate(assigns):
        p = multiprocessing.Process(target= work, args=(i, *assign, ))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
