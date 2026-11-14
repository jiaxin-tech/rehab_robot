"""Inspect explicitly named TwinCAT files without contacting a runtime or device.

XML names and values below are file evidence only. They neither establish the
active hardware identity nor verify the live PDO layout.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import stat
import xml.etree.ElementTree as ET

DEFAULT_CURRENT_CONFIG_PATH = Path("C:/TwinCAT/3.1/Boot/CurrentConfig.xml")
TARGET_INDICES = {0x6041, 0x603F, 0x3C1A}
DEVICE_TAGS = {"device", "iodevice", "ethercatdevice", "box", "iobox", "slave", "ethercatslave"}
IDENTITY_CONTAINERS = {"info", "identity", "deviceinfo", "boxinfo", "ethercat", "identification",
                       "slaveinfo", "type", "vendor", "manufacturer", "product", "revision"}
IDENTITY_FIELDS = {
    "name": "name", "typename": "type", "type": "type", "model": "type", "modelname": "type",
    "vendor": "vendor", "vendorid": "vendor", "manufacturer": "vendor_name",
    "manufacturerid": "vendor", "manufacturername": "vendor_name",
    "product": "product", "productcode": "product", "productid": "product",
    "revision": "revision", "revisionno": "revision", "revisionnumber": "revision",
}
PDO_FIELDS = {"index": "Index", "objectindex": "Index", "subindex": "SubIndex",
              "subidx": "SubIndex", "bitoffs": "BitOffs", "bitoffset": "BitOffs",
              "bitsize": "BitSize", "bitlength": "BitSize", "bitlen": "BitSize"}


def _local(name):
    return str(name).rsplit("}", 1)[-1].split(":")[-1]


def _key(name):
    return "".join(character for character in _local(name).casefold() if character.isalnum())


def _file_snapshot(path, *, retain_bytes=False):
    """Hash one named regular file, returning errors as data and never searching."""
    result = dict(path=str(path), exists=None, is_file=None, size_bytes=None, mtime_ns=None,
                  mtime_utc=None, sha256=None, read_status="NOT_READ", errors=[],
                  metadata_changed_during_read=None)
    try:
        path = Path(path).expanduser().resolve()
        result["path"] = str(path)
        before = path.stat()
        result.update(exists=True, is_file=stat.S_ISREG(before.st_mode), size_bytes=before.st_size,
                      mtime_ns=before.st_mtime_ns,
                      mtime_utc=datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat())
        if not result["is_file"]:
            result["read_status"] = "NOT_A_REGULAR_FILE"
            return result, None
        digest, chunks, bytes_read = hashlib.sha256(), [], 0
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(512 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                bytes_read += len(chunk)
                if retain_bytes:
                    chunks.append(chunk)
        result.update(sha256=digest.hexdigest(), bytes_read=bytes_read, read_status="READ")
        after = path.stat()
        result["metadata_changed_during_read"] = (
            (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)
            or bytes_read != before.st_size)
        if result["metadata_changed_during_read"]:
            result["read_status"] = "READ_FILE_CHANGED"
        return result, b"".join(chunks) if retain_bytes else None
    except FileNotFoundError as exc:
        if result["exists"] is None:
            result["exists"] = False
        result.update(read_status="MISSING" if result["sha256"] is None else "READ_STAT_FAILED")
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        result["read_status"] = "ERROR"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result, None


def _locations(root):
    paths, parents, nodes = {root: "/" + _local(root.tag) + "[1]"}, {}, []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        counts, children = {}, list(node)
        for child in children:
            name = _local(child.tag)
            counts[name] = counts.get(name, 0) + 1
            paths[child] = paths[node] + f"/{name}[{counts[name]}]"
            parents[child] = node
        stack.extend(reversed(children))
    return nodes, paths, parents


def _value(name, value, path, kind):
    return dict(xml_name=name, value=value, xml_path=path, source=kind)


def _identity_kind(name, container):
    key, container = _key(name), _key(container)
    if key == "name" and container in ("vendor", "manufacturer", "product"):
        return "vendor_name" if container != "product" else "product_name"
    if key == "id" and container in ("vendor", "manufacturer"):
        return "vendor"
    if key == "code" and container == "product":
        return "product"
    if key in ("no", "number") and container == "revision":
        return "revision"
    return IDENTITY_FIELDS.get(key)


def _device_identity(device, paths):
    fields, stack = [], [(device, None)]
    while stack:
        node, parent = stack.pop()
        for name, raw in node.attrib.items():
            kind = _identity_kind(name, node.tag)
            if kind and raw.strip():
                fields.append(dict(field=kind, **_value(name, raw, paths[node] + "/@" + _local(name), "attribute")))
        kind = _identity_kind(node.tag, parent.tag if parent is not None else "")
        if node is not device and kind and node.text and node.text.strip():
            fields.append(dict(field=kind, **_value(node.tag, node.text, paths[node], "text")))
        for child in reversed(list(node)):
            key = _key(child.tag)
            if key not in DEVICE_TAGS and (key in IDENTITY_FIELDS or key in IDENTITY_CONTAINERS
                                           or key in ("id", "code", "no", "number")):
                stack.append((child, node))
    result = dict(xml_tag=device.tag, xml_path=paths[device], identity_fields=fields,
                  active_hardware_identity_verified=False)
    for field in ("name", "type", "vendor", "vendor_name", "product", "product_name", "revision"):
        result[field + "_values"] = list(dict.fromkeys(value["value"] for value in fields if value["field"] == field))
    return result


def _direct_pdo_fields(node, paths):
    fields = {name: [] for name in ("Index", "SubIndex", "BitOffs", "BitSize")}
    for name, raw in node.attrib.items():
        kind = PDO_FIELDS.get(_key(name))
        if kind:
            fields[kind].append(_value(name, raw, paths[node] + "/@" + _local(name), "attribute"))
    for child in node:
        kind = PDO_FIELDS.get(_key(child.tag))
        if kind and child.text is not None:
            fields[kind].append(_value(child.tag, child.text, paths[child], "text"))
    return fields


def _index_match(raw):
    value = raw.strip().casefold()
    try:
        if value.startswith(("#x", "0x")):
            number, basis = int(value[2:], 16), "explicit_hex"
        elif value.startswith("16#"):
            number, basis = int(value[3:], 16), "explicit_hex"
        elif len(value) == 4 and value.upper() in {"6041", "603F", "3C1A"}:
            # An unprefixed digit-only 6041 may also be decimal. This is a
            # lexical search hit, never a verified numerical mapping.
            number, basis = int(value, 16), "unprefixed_target_text"
        elif value.isdecimal():
            number, basis = int(value, 10), "explicit_decimal_digits"
        else:
            return None
        if number in TARGET_INDICES:
            return dict(target_label=f"0x{number:04X}", match_basis=basis,
                        ambiguous_unprefixed_numeric=value == "6041")
    except ValueError:
        pass
    return None


def _xml_evidence(data):
    result = dict(xml_parse=dict(status="NOT_PARSED", error=None), devices=[], target_pdo_entries=[])
    if data is None:
        return result
    try:
        root = ET.fromstring(data)
        nodes, paths, parents = _locations(root)
        result["xml_parse"].update(status="PARSED", root_tag=root.tag, element_count=len(nodes))
        devices = {node for node in nodes if _key(node.tag) in DEVICE_TAGS}
        result["devices"] = [_device_identity(node, paths) for node in nodes if node in devices]
        for node in nodes:
            fields = _direct_pdo_fields(node, paths)
            matches = [dict(index_source=value, **match) for value in fields["Index"]
                       if (match := _index_match(value["value"])) is not None]
            if not matches:
                continue
            enclosing = node
            while enclosing not in devices and enclosing in parents:
                enclosing = parents[enclosing]
            result["target_pdo_entries"].append(dict(xml_tag=node.tag, xml_path=paths[node],
                enclosing_device_path=paths[enclosing] if enclosing in devices else None,
                fields=fields, search_matches=matches, live_mapping_verified=False))
    except (ET.ParseError, ValueError, LookupError, RecursionError) as exc:
        result["xml_parse"].update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        result["devices"], result["target_pdo_entries"] = [], []
    return result


def collect_file_evidence(current_config_path=DEFAULT_CURRENT_CONFIG_PATH, *, known_project_files=()):
    """Return one boot-file snapshot plus optional explicit project-file hashes.

    This function does not enumerate directories, import device clients, edit
    configuration, execute commands, or infer that any supplied file is active.
    """
    current, data = _file_snapshot(current_config_path, retain_bytes=True)
    current.update(_xml_evidence(data))
    projects = []
    for path in known_project_files:
        metadata, _ = _file_snapshot(path)
        metadata.update(file_role="EXPLICIT_PROJECT_REFERENCE", active_configuration_status="NOT_VERIFIED")
        projects.append(metadata)
    return dict(evidence_scope="FILE_CONFIGURATION_ONLY", not_live_mapping_verified=True,
        device_access_performed=False, runtime_configuration_modified=False, directory_scan_performed=False,
        current_config_file=current, known_project_files=projects,
        limitations=["File contents and timestamps do not establish the active runtime or live PDO mapping.",
                     "Device identity fields retain XML strings; an absent or ambiguous model is not inferred.",
                     "PDO matches are XML index-field search hits; offsets and units are not interpreted.",
                     "Project references are hashed only and are not assumed to be the active project."])
