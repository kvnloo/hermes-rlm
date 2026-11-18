"""Hermes plugin entry point for the implementation in rlm/."""


def register(ctx):
    # Defer the package-relative import until Hermes calls the qualified entry
    # point. Test collectors can import the checkout root without a parent.
    from .rlm import register as register_implementation
    return register_implementation(ctx)


__all__ = ["register"]
