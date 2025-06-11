import subprocess
import moderngl as mgl

# 1) Point to your slangc.exe
slangc_path = r"C:\\Users\\PGOSH\\Downloads\\slang-2025.10.2-windows-x86_64\\bin\\slangc.exe"

cmd = [
    slangc_path,
    "shaders/radix_sort.slang",     # your .slang file
    "-lang",        "glsl",
    "-target",      "glsl",         # emit GLSL (not SPIR-V)
    "-profile",     "glsl_450",     # choose a core GLSL version with compute
    "-entry",       "buildDeviceHistogram",
    "-reflection-json", "reflection.json",
    "-o",           "out.glsl",
    "-I",           "shaders",       # include directory, not the .slang file
    "-force-glsl-scalar-layout"
]

subprocess.run(cmd, check=True)
# print("✅ test.slang → out.spv")
# ctx = mgl.create_standalone_context()
# with open('out.glsl') as f:
#     ctx.compute_shader(f.read())