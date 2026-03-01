"""compressor_2.evolution – delta-evolution loop for steering magnitudes."""


def run_evolution(cfg):
    """Branch on optimization_method: DEAP GA or zero-order path."""
    method = getattr(cfg, "optimization_method", "deap")
    if method == "deap":
        from .ga_driver import run_ga_evolution
        run_ga_evolution(cfg)
    else:
        from .optimize_driver import run_zero_order_evolution
        run_zero_order_evolution(cfg)


__all__ = ["run_evolution"]
