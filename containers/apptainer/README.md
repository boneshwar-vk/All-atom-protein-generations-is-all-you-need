# Apptainer / Singularity definitions (HPC)

Mirror the Docker images for clusters without Docker. Build from the Docker images or convert:

```bash
# from a built docker image
apptainer build rfd3.sif docker-daemon://aapg/rfd3:latest
```

`.def` files (`base.def`, `rfd3.def`, `proteinmpnn.def`, `protenix.def`, `opendde.def`,
`rosetta.def`) go here, mirroring `containers/*.Dockerfile`.

**Status:** placeholder — add `.def` files alongside the Dockerfiles at implementation time.
