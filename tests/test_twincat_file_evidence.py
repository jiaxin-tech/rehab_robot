"""Temporary files only; no TwinCAT or device dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.twincat_file_evidence import DEFAULT_CURRENT_CONFIG_PATH, collect_file_evidence


def assert_file_only(result):
    assert result["evidence_scope"] == "FILE_CONFIGURATION_ONLY"
    assert result["not_live_mapping_verified"] is True
    assert result["device_access_performed"] is False
    assert result["runtime_configuration_modified"] is False
    assert result["directory_scan_performed"] is False
    json.dumps(result, allow_nan=False)


def test_missing_file_is_evidence_and_never_scans_directories(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("directory scanning is outside this helper")
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    result = collect_file_evidence(tmp_path / "missing.xml")
    assert_file_only(result)
    current = result["current_config_file"]
    assert current["exists"] is False
    assert current["read_status"] == "MISSING"
    assert current["sha256"] is None
    assert current["xml_parse"]["status"] == "NOT_PARSED"
    assert current["devices"] == current["target_pdo_entries"] == []
    assert result["known_project_files"] == []
    assert DEFAULT_CURRENT_CONFIG_PATH == Path("C:/TwinCAT/3.1/Boot/CurrentConfig.xml")


def test_namespace_identity_and_pdo_fields_preserve_file_strings(tmp_path):
    data = b'''<?xml version="1.0"?>
<TcConfig xmlns="urn:fake-twincat"><Io><Device Name="EtherCAT bus">
  <Box><Name>FAKE DRV-22</Name><Type ProductCode="#x00AB12CD" RevisionNo="#x00000003">FAKE-Model</Type>
    <Info><Vendor><Id>#x00000002</Id><Name>Fake vendor</Name></Vendor></Info>
    <TxPdo><Entry Index="#x6041" SubIndex="00" BitOffs=" 0 " BitSize="16"><Name>Statusword</Name></Entry>
      <Entry><Index> 0x603f </Index><SubIndex>0</SubIndex><BitOffs>128</BitOffs><BitSize>16</BitSize></Entry>
      <Entry><Index>15386</Index><SubIndex>01</SubIndex><BitOffs>160</BitOffs><BitSize>32</BitSize></Entry>
      <Entry Index="#x1A00"/>
    </TxPdo>
  </Box>
</Device></Io></TcConfig>'''
    path = tmp_path / "CurrentConfig.xml"
    path.write_bytes(data)
    original_stat = path.stat()
    result = collect_file_evidence(path)
    assert_file_only(result)
    current = result["current_config_file"]
    assert current["exists"] and current["is_file"]
    assert current["size_bytes"] == len(data)
    assert current["sha256"] == hashlib.sha256(data).hexdigest()
    assert current["mtime_ns"] == original_stat.st_mtime_ns
    assert current["mtime_utc"].endswith("+00:00")
    assert current["metadata_changed_during_read"] is False
    assert current["xml_parse"]["status"] == "PARSED"
    device, box = current["devices"]
    assert device["name_values"] == ["EtherCAT bus"]
    assert box["name_values"] == ["FAKE DRV-22"]
    assert box["type_values"] == ["FAKE-Model"]
    assert box["vendor_values"] == ["#x00000002"]
    assert box["vendor_name_values"] == ["Fake vendor"]
    assert box["product_values"] == ["#x00AB12CD"]
    assert box["revision_values"] == ["#x00000003"]
    assert "Statusword" not in box["name_values"]
    assert device["product_values"] == []  # Do not attach a child box identity to its bus.
    entries = current["target_pdo_entries"]
    assert len(entries) == 3
    assert [entry["search_matches"][0]["target_label"] for entry in entries] == ["0x6041", "0x603F", "0x3C1A"]
    assert entries[0]["fields"]["BitOffs"][0]["value"] == " 0 "
    assert entries[1]["fields"]["Index"][0]["value"] == " 0x603f "
    assert entries[2]["fields"]["Index"][0]["value"] == "15386"
    assert entries[2]["fields"]["SubIndex"][0]["value"] == "01"
    assert len({entry["xml_path"] for entry in entries}) == 3
    assert all(entry["enclosing_device_path"] == box["xml_path"] for entry in entries)
    assert all(entry["live_mapping_verified"] is False for entry in entries)
    assert path.read_bytes() == data
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_target_entry_without_device_does_not_invent_a_model(tmp_path):
    path = tmp_path / "unfamiliar.xml"
    path.write_text('<Unfamiliar><Entry Index="#x6041" BitOffs="unknown"/></Unfamiliar>', encoding="utf-8")
    current = collect_file_evidence(path)["current_config_file"]
    assert current["devices"] == []
    entry, = current["target_pdo_entries"]
    assert entry["enclosing_device_path"] is None
    assert entry["fields"]["SubIndex"] == []
    assert entry["fields"]["BitOffs"][0]["value"] == "unknown"


@pytest.mark.parametrize("value, label, ambiguous", [("#x6041", "0x6041", False),
    ("0x603F", "0x603F", False), ("16#3C1A", "0x3C1A", False),
    ("24641", "0x6041", False), ("3C1A", "0x3C1A", False), ("6041", "0x6041", True)])
def test_index_search_matches_are_explicit_and_keep_ambiguity(tmp_path, value, label, ambiguous):
    path = tmp_path / "indices.xml"
    path.write_text(f'<Root><Entry Index="{value}"/></Root>', encoding="utf-8")
    entry, = collect_file_evidence(path)["current_config_file"]["target_pdo_entries"]
    match, = entry["search_matches"]
    assert match["target_label"] == label
    assert match["ambiguous_unprefixed_numeric"] is ambiguous
    assert match["index_source"]["value"] == value


def test_malformed_xml_preserves_metadata_and_hash(tmp_path):
    path = tmp_path / "bad.xml"
    data = b'<Root><Box Name="fake"></Root>'
    path.write_bytes(data)
    result = collect_file_evidence(path)
    assert_file_only(result)
    current = result["current_config_file"]
    assert current["read_status"] == "READ"
    assert current["sha256"] == hashlib.sha256(data).hexdigest()
    assert current["xml_parse"]["status"] == "ERROR"
    assert current["devices"] == current["target_pdo_entries"] == []


def test_permission_error_does_not_crash_or_claim_file_missing(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.xml"
    path.write_text("<Root/>", encoding="utf-8")
    original_open = Path.open

    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("fake test denial")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    current = collect_file_evidence(path)["current_config_file"]
    assert current["exists"] is True
    assert current["read_status"] == "ERROR"
    assert current["sha256"] is None
    assert "PermissionError" in current["errors"][0]
    assert current["xml_parse"]["status"] == "NOT_PARSED"


def test_directory_is_not_read_as_a_file(tmp_path):
    current = collect_file_evidence(tmp_path)["current_config_file"]
    assert current["exists"] and current["is_file"] is False
    assert current["read_status"] == "NOT_A_REGULAR_FILE"
    assert current["xml_parse"]["status"] == "NOT_PARSED"


def test_only_explicit_project_files_are_hashed_and_never_assumed_active(tmp_path):
    project = tmp_path / "known.tsproj"
    data = b"not XML; project references must only be hashed"
    project.write_bytes(data)
    result = collect_file_evidence(tmp_path / "missing_current.xml", known_project_files=[project, tmp_path / "missing.tsproj"])
    assert_file_only(result)
    present, missing = result["known_project_files"]
    assert present["sha256"] == hashlib.sha256(data).hexdigest()
    assert present["active_configuration_status"] == missing["active_configuration_status"] == "NOT_VERIFIED"
    assert "xml_parse" not in present
    assert missing["exists"] is False


def test_xml_external_entity_cannot_read_another_file(tmp_path):
    private_file = tmp_path / "unrelated.txt"
    marker = "must-not-be-read-or-included"
    private_file.write_text(marker, encoding="utf-8")
    path = tmp_path / "external.xml"
    path.write_text(f'<!DOCTYPE Root [<!ENTITY ext SYSTEM "{private_file.as_uri()}">]><Root><Box><Name>&ext;</Name></Box></Root>', encoding="utf-8")
    result = collect_file_evidence(path)
    assert result["current_config_file"]["xml_parse"]["status"] == "ERROR"
    assert marker not in json.dumps(result)
