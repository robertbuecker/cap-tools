from pathlib import Path
import re
import shutil
import subprocess
import sys
import warnings

from cx_Freeze import Executable, setup
from cx_Freeze.command.build_exe import build_exe


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "dist-cx" / "cap_tools"
EXPECTED_ENV_NAMES = {"cap-tools-pip", "cap_tools_pip"}
DESCRIBED_VERSION_RE = re.compile(r"^(?P<tag>.+)-(?P<commits>\d+)-g[0-9a-f]+$")
VERSION_TAG_RE = re.compile(
    r"^v?"
    r"(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:\.(0|[1-9]\d*))?"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"$"
)


def _assert_expected_env() -> None:
    env_name = Path(sys.prefix).name.lower()
    if env_name not in EXPECTED_ENV_NAMES:
        expected = " or ".join(sorted(EXPECTED_ENV_NAMES))
        raise SystemExit(
            f"cx_Freeze distribution must be built from {expected}; "
            f"current Python is {sys.executable}"
        )


def _conda_library_bin(filename: str) -> tuple[str, str]:
    source = Path(sys.prefix) / "Library" / "bin" / filename
    if not source.exists():
        raise FileNotFoundError(source)
    return str(source), f"lib/{filename}"


def _git_describe() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"Could not determine version from git describe: {exc}") from exc


def _is_version_tag(version: str) -> bool:
    return VERSION_TAG_RE.fullmatch(version) is not None


def _version_tag_from_git_describe() -> str:
    described_version = _git_describe()
    match = DESCRIBED_VERSION_RE.fullmatch(described_version)
    if match and _is_version_tag(match.group("tag")):
        version_tag = match.group("tag")
        warnings.warn(
            f"git describe returned '{described_version}', which is not a clean "
            f"version tag; using previous tag '{version_tag}' for cx_Freeze.",
            stacklevel=2,
        )
        return version_tag

    if _is_version_tag(described_version):
        return described_version

    raise SystemExit(
        f"git describe returned '{described_version}', which is not a supported "
        "semantic version tag or tag-plus-commit description."
    )


def _setup_version(version_tag: str) -> str:
    return version_tag[1:] if version_tag.startswith("v") else version_tag


def _write_version_file(version_tag: str) -> None:
    (ROOT / "version.txt").write_text(f"{version_tag}\n", encoding="ascii")


_assert_expected_env()
PROGRAM_VERSION_TAG = _version_tag_from_git_describe()
PROGRAM_SETUP_VERSION = _setup_version(PROGRAM_VERSION_TAG)

if "--print-version-tag" in sys.argv:
    print(PROGRAM_VERSION_TAG)
    raise SystemExit

include_files = [
    ("version.txt", "version.txt"),
    ("cell_tool_icon.ico", "cell_tool_icon.ico"),
    ("calibrate_dd_icon.ico", "calibrate_dd_icon.ico"),
    _conda_library_bin("tcl86t.dll"),
    _conda_library_bin("tk86t.dll"),
]

includes = [
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_pdf",
    "scipy.cluster.hierarchy",
    "scipy.spatial.distance",
    "skimage.filters.ridges",
    "skimage.registration._phase_cross_correlation",
    "sklearn.decomposition._pca",
    "sklearn.preprocessing._data",
]

excludes = [
    "backports",
    "colorama",
    "comm",
    "debugpy",
    "greenlet",
    "IPython",
    "ipykernel",
    "jaraco",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "more_itertools",
    "notebook",
    "psutil",
    "pytest",
    "setuptools_scm",
    "sqlalchemy",
    "sphinx",
    "tornado",
    "traitlets",
    "zmq",
    "tkinter.test",
    "matplotlib.backends.backend_gtk3",
    "matplotlib.backends.backend_gtk3agg",
    "matplotlib.backends.backend_gtk3cairo",
    "matplotlib.backends.backend_gtk4",
    "matplotlib.backends.backend_gtk4agg",
    "matplotlib.backends.backend_gtk4cairo",
    "matplotlib.backends.backend_macosx",
    "matplotlib.backends.backend_nbagg",
    "matplotlib.backends.backend_qt",
    "matplotlib.backends.backend_qt5",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qt5cairo",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qtcairo",
    "matplotlib.backends.backend_webagg",
    "matplotlib.backends.backend_webagg_core",
    "matplotlib.backends.backend_wx",
    "matplotlib.backends.backend_wxagg",
    "matplotlib.backends.backend_wxcairo",
    "matplotlib.backends.qt_compat",
    "matplotlib.tests",
    "matplotlib.testing",
    "numpy.tests",
    "pandas.tests",
    "scipy.tests",
    "sklearn.tests",
    "skimage.color",
    "skimage.draw",
    "skimage.future",
    "skimage.graph",
    "skimage.io",
    "skimage.metrics",
    "skimage.morphology",
    "skimage.restoration",
    "skimage.segmentation",
    "skimage.viewer",
    "skimage._shared.tests",
    "skimage.feature.tests",
    "skimage.filters.tests",
    "skimage.registration.tests",
]

build_exe_options = {
    "build_exe": str(BUILD_DIR),
    "bin_path_excludes": [
        r"C:\Program Files\Eclipse Adoptium\jre-8.0.452.9-hotspot\bin",
    ],
    "include_files": include_files,
    "includes": includes,
    "packages": ["cap_tools"],
    "excludes": excludes,
    "include_msvcr": True,
    "optimize": 1,
}


class BuildExe(build_exe):
    prune_top_level = {
        "backports",
        "colorama",
        "comm",
        "debugpy",
        "greenlet",
        "ipykernel",
        "jaraco",
        "more_itertools",
        "psutil",
        "sqlalchemy",
        "tornado",
        "traitlets",
        "zmq",
    }
    prune_dir_names = {"__pycache__", "tests"}
    prune_suffixes = {".c", ".cpp", ".h", ".pxd"}

    def run(self) -> None:
        super().run()
        self._prune_build_tree()

    def _prune_build_tree(self) -> None:
        lib_dir = Path(self.build_exe) / "lib"
        if not lib_dir.exists():
            return

        for name in self.prune_top_level:
            shutil.rmtree(lib_dir / name, ignore_errors=True)

        for directory in sorted(lib_dir.rglob("*"), reverse=True):
            if directory.is_dir() and directory.name in self.prune_dir_names:
                shutil.rmtree(directory, ignore_errors=True)

        for file_path in lib_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.prune_suffixes:
                file_path.unlink()

executables = [
    Executable(
        "cell_tool.py",
        base="Win32GUI",
        target_name="cell_tool.exe",
        icon="cell_tool_icon.ico",
    ),
    Executable(
        "calibrate_dd.py",
        base="Win32GUI",
        target_name="calibrate_dd.exe",
        icon="calibrate_dd_icon.ico",
    ),
    Executable(
        "generate_learning_set.py",
        base=None,
        target_name="generate_learning_set.exe",
    ),
]

def main() -> None:
    _write_version_file(PROGRAM_VERSION_TAG)
    setup(
        name="cap_tools",
        version=PROGRAM_SETUP_VERSION,
        description="Electron diffraction analysis tools",
        cmdclass={"build_exe": BuildExe},
        options={"build_exe": build_exe_options},
        executables=executables,
    )


if __name__ == "__main__":
    main()
