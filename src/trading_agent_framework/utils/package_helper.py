"""Installed-package metadata helpers."""

from importlib.metadata import PackageNotFoundError, version


def get_version(package_name: str, fallback: str = "0.0.0.dev0") -> str:
    """Return an installed package's version.

    Args:
        package_name: Distribution name.
        fallback: Returned when the package is not installed.

    Returns:
        The installed version, or `fallback`.
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return fallback
