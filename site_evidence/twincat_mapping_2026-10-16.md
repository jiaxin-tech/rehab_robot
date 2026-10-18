# TwinCAT mapping evidence — 2026-10-16

Status: file-level evidence plus a live read-only ADS diagnostic collected; **not yet accepted as an on-site human review**.

## Sources

- Active boot configuration: `C:/TwinCAT/3.1/Boot/CurrentConfig.xml`
  - modified: 2026-09-29 18:53:20 local time
  - SHA-256: `C5D10C97C6D4C131A3FF6871A509BC71A7A559DFFF585A44DFF70F89A2F5934C`
- Newer project file: `C:/Users/liumai/Documents/TcXaeShell/TwinCAT 机器腿/TwinCAT 机器腿/TwinCAT 机器腿.tsproj`
  - modified: 2026-09-29 18:59:02 local time
  - SHA-256: `117F192C5D66BEF7FFB8F9F30378FE7FB918C3B28EA2A6B9D1CE973439685250`

Because the project file is newer than the active boot configuration, the project must still be compared with the currently active runtime in TwinCAT before `mapping_reviewed` can be set to `true`.

## Facts present in both files

- Four EtherCAT drives are configured.
- ADS port: `11`.
- Index group: `50593808` (`0x03040210`).
- Each selected transmit PDO contains, in order:
  - `0x6041` Statusword, UINT16
  - `0x6064` Position Actual Value, DINT32
  - `0x606C` Velocity Actual Value, DINT32
  - `0x6077` Torque Actual Value, INT16
  - `0x60F4` Following Error Actual Value, DINT32
  - `0x603F` Error Code, UINT16
  - `0x6061` Mode of Operation Display, SINT8
  - `0x3C1A` Actual TorqueClose Value, REAL32
- Selected transmit PDO size per axis is 23 bytes.
- Axis statusword offsets are:
  - axis 1: `0x8000009F` (`2147483807`)
  - axis 2: `0x800000B6` (`+23`)
  - axis 3: `0x800000CD` (`+46`)
  - axis 4: `0x800000E4` (`+69`)
- `0x3C1A` offsets are:
  - axis 1: `0x800000B2`
  - axis 2: `0x800000C9`
  - axis 3: `0x800000E0`
  - axis 4: `0x800000F7`
- The project links the EtherCAT `WcState/InputToggle` bit to each NC axis state input.
- NC SAF task cycle time is `20000` in TwinCAT units, consistent with the historical 2 ms / 500 Hz task setting.

## Items requiring live review

- The Windows TwinCAT registry records the local AMS Router Net ID as `10.20.4.89.1.1`; this is the ADS target used with port `11` in `config/twincat_leg_capture.json`.
- The project and active boot file identify the EtherCAT device as `10.20.4.89.3.1`. This is a separate EtherCAT-device AMS address, so it is not a conflict with the local AMS Router target.
- The remaining technical item is the physical signedness/semantics of `0x6077`; the static live snapshot returned raw zero for all four axes, so it cannot establish the sign convention. Do not set `mapping_reviewed=true` until a responsible reviewer accepts the live evidence and the signedness check.

## Live read-only ADS evidence

Source report: `results/lab_visit/twincat_readonly_mapping_20261016.json`.

- Target `10.20.4.89.1.1:11`, index group `50593808`, index offset `2147483807`.
- 200/200 reads succeeded; every read returned exactly 1366 bytes.
- All four 23-byte axis records decoded at offsets `0/23/46/69`.
- Byte 1365 bit 4 produced both values 0 and 1, with 92 observed transitions; byte 1365 changed in the live process image.
- Query duration: mean 1.165 ms, p95 2.053 ms, maximum 2.193 ms.
- Last snapshot showed all four axes `operation_enabled=false`, `fault=false`, error `0x0000`, mode display 8.
- Last `0x3C1A` values were axis1 `-22.6305`, axis2 `-2.2704`, axis3 `0.2777`, axis4 `-1.7455` N·m. These are observations, not zero offsets or approved limits.
- `0x6077` raw bits were zero in the snapshot; signedness therefore remains unverified.

## Actions deliberately not taken

- No TwinCAT activation, download, mode switch, online write, CoE write, or PLC write.
- No robot or dummy-leg enable, reset, brake release, or motion command.
- TwinCAT GUI automation was attempted only to inspect the live mapping, but the Windows computer-control helper failed to initialize (`trusted Node process exited unexpectedly`). No GUI action was issued. The later ADS diagnostic confirmed the live read span and toggle location; human review and `0x6077` sign semantics remain pending.
- The later live diagnostic used only `AdsPortOpenEx`, `AdsSyncSetTimeoutEx`, `AdsSyncReadReqEx2`, and `AdsPortCloseEx`; no ADS write symbol was exposed by the diagnostic.
