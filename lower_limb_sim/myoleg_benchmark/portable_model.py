"""Load frozen MyoLeg XML with local MyoSuite assets, without editing the XML.

``asset_root`` (or ``MYOSUITE_ASSET_ROOT``) is the directory containing the
MyoSuite ``scene`` and ``meshes`` directories, usually ``simhive/myo_sim``.
Only asset file locations are rewritten in memory; all model parameters are
passed unchanged to MuJoCo. The default asset cache contains files extracted
from the official MyoSuite 2.12.2 wheel, not an installed MyoSuite environment.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml"
)
ASSET_VERSION = "2.12.2"
ASSET_SOURCE_URL = (
    "https://files.pythonhosted.org/packages/b7/35/"
    "755f9ba4a593a91f5453473cde0eea256a0092adf1e2eff57a01d8675517/"
    "myosuite-2.12.2-py3-none-any.whl"
)
DEFAULT_ASSET_ROOT = (
    PROJECT_ROOT / ".cache/myosuite-assets" / ASSET_VERSION
    / "myosuite/simhive/myo_sim"
)
_ASSET_MARKER = "/myosuite/simhive/myo_sim/"


def _asset_location(asset_root: str | Path | None) -> tuple[Path, dict]:
    requested = asset_root or os.environ.get("MYOSUITE_ASSET_ROOT")
    if requested is not None:
        root = Path(requested).expanduser().resolve()
        source = {"selection": "explicit_or_environment", "myosuite_version": None}
    elif DEFAULT_ASSET_ROOT.is_dir():
        root = DEFAULT_ASSET_ROOT
        source = {
            "selection": "official_wheel_cache",
            "myosuite_version": ASSET_VERSION,
            "source_url": ASSET_SOURCE_URL,
        }
    else:
        try:
            distribution = importlib.metadata.distribution("myosuite")
        except importlib.metadata.PackageNotFoundError as exc:
            raise FileNotFoundError(
                "MyoSuite assets are missing. Extract the scene and meshes "
                f"directories from MyoSuite {ASSET_VERSION} into "
                f"{DEFAULT_ASSET_ROOT}, or set MYOSUITE_ASSET_ROOT / asset_root "
                "to an existing simhive/myo_sim directory."
            ) from exc
        root = Path(distribution.locate_file("myosuite/simhive/myo_sim")).resolve()
        source = {
            "selection": "installed_distribution",
            "myosuite_version": distribution.version,
        }
    if not root.is_dir():
        raise FileNotFoundError(f"MyoSuite asset directory does not exist: {root}")
    return root, source


def load_model_with_provenance(
    model_path: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> tuple[mujoco.MjModel, dict]:
    """Return a compiled model and the source locations used to load it.

    The source XML stays untouched. Unrecognized file references are resolved
    relative to the source XML, so missing files remain errors rather than
    silently removing any mesh, texture, or mechanical component.
    """
    source_path = Path(model_path or DEFAULT_MODEL_PATH).expanduser().resolve()
    document = ET.parse(source_path)
    root, provenance = _asset_location(asset_root)
    relocated = []
    for element in document.iter():
        filename = element.get("file")
        if filename is None:
            continue
        normalized = filename.replace("\\", "/")
        if _ASSET_MARKER in normalized:
            relative = normalized.split(_ASSET_MARKER, 1)[1]
            target = root / relative
            relocated.append(relative)
        else:
            target = source_path.parent / filename
        if not target.is_file():
            raise FileNotFoundError(f"Missing MyoLeg {element.tag} asset: {target}")
        element.set("file", str(target.resolve()))
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(document.getroot(), encoding="unicode")
    )
    return model, {
        **provenance,
        "model_path": str(source_path),
        "asset_root": str(root),
        "relocated_asset_count": len(relocated),
        "asset_files": relocated,
        "mujoco_version": mujoco.__version__,
    }


def load_model(
    model_path: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> mujoco.MjModel:
    """Compile the frozen model using local assets; see module docstring."""
    model, _ = load_model_with_provenance(model_path=model_path, asset_root=asset_root)
    return model
