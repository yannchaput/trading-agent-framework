from importlib.metadata import PackageNotFoundError, version


def get_version(package_name: str, fallback: str = "0.0.0.dev0") -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return fallback
