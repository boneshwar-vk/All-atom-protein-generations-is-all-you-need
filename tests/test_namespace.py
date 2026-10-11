"""Cross-package smoke test: the whole workspace imports under the shared namespace."""


def test_all_packages_import():
    import aapg.core  # noqa: F401
    import aapg.ellipsoid  # noqa: F401
    import aapg.grpo  # noqa: F401
    import aapg.mpnn_iter  # noqa: F401
    import aapg.pipelines  # noqa: F401
    import aapg.toolkit  # noqa: F401
    import aapg.validation  # noqa: F401
    import proteinmpnn  # noqa: F401
    import rfdiffusion3  # noqa: F401
