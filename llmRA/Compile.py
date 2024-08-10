import os
import sys
import time
import subprocess
from .Opt.optimizer import onlineOPT, onlineGAOPT
from .utils import *

def compile_a_program(program, file_path, first=True, cfgid = -1):
    path = program['path']
    compiler = program['compiler']
    options = program['options']

    # First time extract information
    if first:
        additional = "-mllvm --regalloc=grad -mllvm --dump-info -mllvm -ra-score-log=%s"%file_path
    else:
        assert cfgid != -1
        additional = "-mllvm --regalloc=grad -mllvm -ra-score-log=%s -mllvm -load-cfg-id=%d"%(file_path, cfgid)
    print(compiler + " " + path + " " + options +  " " + additional)

    command = [compiler, path] + options.split(" ") + additional.split(" ")
    try:
        result = subprocess.run(command, check=True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for {path}")
        print(e.stderr.decode())
        assert 0


def filter_data(data):
    to_rm = []
    for wlkey in data:
        intrv_path = ".%s_intervals"%wlkey
        if not os.path.exists(intrv_path):
            to_rm.append(wlkey)
    for wlkey in to_rm:
        del data[wlkey]
    return data

def measure(project_name, program, file_path, all_cfgs):
    print("Measuring...")
    res = {}
    for wlkey in all_cfgs:
        res[wlkey] = []
    count = 0
    for wlkey, cfgs in all_cfgs.items():
        for i, cfg in enumerate(cfgs):
            cfgname = ".%s_result_%d"%(wlkey, i)
            write_list_to_file(cfgname, cfg)
        count = len(cfgs)
    for i in range(count):
        remove_project_file(project_name)
        compile_a_program(program, file_path, False, i)
        print('*', end=' ')
        sys.stdout.flush() 
        if (i + 1) % 10 == 0:
            print()
        data = read_project_file(project_name)
        for wlkey in all_cfgs:
            res[wlkey].append(data[wlkey])
   #for wlkey, cfgs in all_cfgs.items():
   #    for i, cfg in enumerate(cfgs):
   #        cfgname = ".%s_result_%d"%(wlkey, i)
   #        os.remove(cfgname)
    return res 

def compile_programs_online(programs, project_name):
    remove_project_file(project_name)
    file_path = f".{project_name}"
    for program in programs:
        if 'path' not in program or 'compiler' not in program or 'options' not in program:
            print(f"Invalid program entry: {program}")
            continue

        s = time.time()
        compile_a_program(program, file_path, True)
        print("One trial compilation time : ", time.time() - s)
        data = read_project_file(project_name)
        data = filter_data(data)
        print(data)

        # Iterative compilation for best allocation strategy.
        opt = onlineGAOPT(list(data.keys()))
        for i in range(1):
            all_cfgs = {}
           #for workload_key in data:
            for workload_key in ["main"]:
                cfgs = opt.explore(workload_key)
                all_cfgs[workload_key] = cfgs
                assert 0 
            scores = measure(project_name, program, file_path, all_cfgs)
            print(scores)
            opt.update(all_cfgs, scores)
        opt.dump()
       #remove_additional_files(list(data.keys()))
        


