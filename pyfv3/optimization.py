from ndsl import Backend, OptimizationConfig


def get_optimization_config(backend: Backend) -> OptimizationConfig:
    if backend.is_gpu_backend():
        return OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=False,
            ),
            gpu=OptimizationConfig.GPU(common_gpu_xforms=False),
        )

    return OptimizationConfig(
        stree=OptimizationConfig.Tree(
            enabled=False,
        ),
    )
