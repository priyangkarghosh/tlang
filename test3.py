from shader_processor import ShaderProcessor


src = """
#include<test2>
[include(test, test1, test3)]
[extend('subgroup')]

[program('main', vert='test', frag='test2')]
void unusedHelper() {
    // should not get linked to any attributes
}

[shader('compute'), numthreads(64, 1, 1)]
void mainKernel() {
    uint sum = 0;

    [unroll]
    for (int i = 0; i < 10; ++i) {
        sum += i;
    }
}

[shader('compute'), numthreads(64, 1, 1)]
void helperKernel() {
    uint prod = 1;

    for (int i = 1; i <= 5; ++i) {
        prod *= i;
    }
}
"""

sp = ShaderProcessor('test', src)
print(sp.src_map)
print("============================")
print(sp.glob_attrs)