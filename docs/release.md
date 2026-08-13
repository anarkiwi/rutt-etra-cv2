# Releasing

`.github/workflows/release.yml` publishes a multi-architecture image to Docker Hub.
It builds the `test` target and runs the whole suite first; a red suite blocks the
push.

## Repository settings

| Kind | Name | Value |
|---|---|---|
| Secret | `DOCKERHUB_USERNAME` | Docker Hub account name |
| Secret | `DOCKERHUB_TOKEN` | access token from Docker Hub, **Account Settings → Personal access tokens**, scope *Read & Write* |
| Variable (optional) | `DOCKERHUB_IMAGE` | full image name, e.g. `myorg/rutt-etra-cv2` |

Without `DOCKERHUB_IMAGE` the image is named after the GitHub owner:
`<owner>/rutt-etra-cv2`. Set the variable if the Docker Hub namespace differs, or if
the owner name is not already lowercase — Docker Hub rejects uppercase.

Use a token, never the account password, and prefer a token scoped to the one
repository if the account holds anything else.

## Publishing

```sh
git tag v0.2.0
git push origin v0.2.0
```

Tags matching `v*` publish `0.2.0`, `0.2` and `latest`. The workflow can also be run
by hand from the Actions tab, which publishes whatever tag you name and leaves
`latest` alone.

## Image layout

The Dockerfile has four stages. `deps` installs runtime dependencies into a venv,
`devdeps` adds the test tooling, `test` carries the suite and defaults to `pytest`,
and `runtime` is what ships — venv plus package, no tests, no docs.

```sh
docker run --rm -v "$PWD:/data" <image> \
  rutt-etra.py /data/in.mp4 --no-monitor --outfile /data/out.avi
docker run --rm -v "$PWD:/data" <image> \
  rutt-scope.py /data/scope.wav --outfile /data/scope-view.avi
```

The entry point is `python`, so the first argument selects the tool. Pass
`--no-monitor`: the image has no display, and `opencv-python-headless` cannot open
a preview window.
