# xCoreSDK Windows runtime

This directory contains the Rokae xCoreSDK Python 0.7.0 Windows runtime used by
`hardware/windows/rokae_xcore.py`:

- `xCoreSDK_python.cp312-win_amd64.pyd`: CPython 3.12, 64-bit Windows extension
- `xCoreSDK.dll`: native runtime dependency
- `xCoreSDK_python/`: vendor type declarations for the base, realtime, model,
  planner, and utility APIs
- `CHANGELOG.md`: vendor release history and controller compatibility

The adapter verifies `BaseRobot.sdkVersion() == "0.7.0"` before connecting.
xCoreSDK 0.7.0 requires an xCore controller at version 3.2 or newer.
