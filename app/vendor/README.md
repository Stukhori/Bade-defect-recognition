# Vendored Ultralytics deployment wheel

`ultralytics-8.3.150-py3-none-any.whl` is a reproducible deployment build of the
official Ultralytics 8.3.150 source distribution. Its Python package payload is
byte-for-byte identical to the official 8.3.150 wheel. The sole source metadata
change replaces `opencv-python>=4.6.0` with
`opencv-python-headless==4.11.0.86`, preventing pip from installing two OpenCV
distributions on the headless Streamlit server.

- Upstream project and license: Ultralytics, AGPL-3.0
- Official source distribution SHA-256: `ffeb6ed91b2bb91eafd89580cb51a7b01b9e3a04a64b91c77e68bfa41d14ed58`
- Official wheel SHA-256 used for payload comparison: `4bd99f2c33d8372c67813eb150a4d8a755cc49f159ab9382b0650799f724d886`
- Vendored wheel SHA-256: `1827a8504ef9c70b3285069b70ce0bc92f434934a06d0c550601d0a5fc2455bb`
- Deterministic build environment: Python 3.11, setuptools 84.0.0, wheel
  0.48.0, `SOURCE_DATE_EPOCH=315532800`, `PYTHONHASHSEED=0`

Build the wheel from the official 8.3.150 source after applying only the
dependency substitution above. Two clean builds produced the recorded wheel
hash. This wheel does not alter BladeScope model execution or scientific
artifacts.
