# aapg-toolkit

Adapters to **external** protein tools, each implementing an `aapg.core.scoring` interface and
registering itself by name. Tools run in their own containers (see `containers/`) and are invoked
via subprocess, so heavy/conflicting deps never pollute the main environment.

Planned adapters (placeholders): `rosetta`, `protenix` (v2), `opendde`, `sequence_models` (ESM,
...), `tmalign`, `foldseek`.

**Status:** placeholder stubs only.
