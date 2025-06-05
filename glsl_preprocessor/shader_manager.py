from moderngl import ComputeShader, Context, Program
import moderngl


class ShaderManager:
    def __init__(
        self, 
        ctx: Context, 
        sources: dict[str, str], 
        entries: dict[str, list[dict]], 
        programs: dict[str, dict[str, dict]]
    ) -> None:
        self._ctx = ctx
        self.shaders: dict[str, str] = {}
        self.programs: dict[str, Program] = {}
        self.kernels: dict[str, ComputeShader] = {}

        self.shaders: dict[str, str] = {}
        for src_name, src in sources.items():
            prefix = f"{src_name}/"
            for ep in entries.get(src_name, []):
                shader_name = prefix + ep['name']
                shader_src = src + "\n\n" + ep['full_function']
                self.shaders[shader_name] = shader_src
                print(f'Created shader {shader_name}')

                # check if this is a compute shader
                # -> if so, then compile it
                try: self.kernels[shader_name] = self._ctx.compute_shader(shader_src)
                except moderngl.Error as e: 
                    print(f"[ERROR] Kernel compile failed for {shader_name}: {e}")
            
            for program_name, stages in programs[src_name].items():
                try:
                    self.programs[program_name] = self._ctx.program(
                        vertex_shader=stages['vert'],
                        fragment_shader=stages['frag'] or None,
                        geometry_shader=stages['geom'] or None,
                        tess_control_shader=stages['tess_ctrl'] or None,
                        tess_evaluation_shader=stages['tess_eval'] or None,
                        # look into varyings and fragment outputs
                    )
                except Exception as e:
                    print(f"[ERROR] Program compile failed for {program_name}: {e}")
