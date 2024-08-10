import llmRA
programs_to_compile = [
    {
        'path': '/Users/bijun/work/ra/tests/llvm-test-suite/SingleSource/Benchmarks/Polybench/stencils/heat-3d/heat-3d.c',
       #'path': '/Users/bijun/work/ra/tests/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/lpbench.c',
        'compiler': 'clang',
        'options': '-target x86_64-pc-linux-gnu  \
                    -I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include \
                    -I /Users/bijun/work/ra/tests/llvm-test-suite/SingleSource/Benchmarks/Polybench/utilities -DPOLYBENCH_DUMP_ARRAYS -DFP_ABSTOLERANCE=1e-5 \
                    -S -O3 -o lcm.s -mllvm -no-split-loop'
    },
]

llmRA.compile_programs_online(programs_to_compile, "my")


