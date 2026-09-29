# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Highlights

- Added a reproducible release pipeline for the Python package.
- Added SHA256 manifests for release artifacts.
- Added a user-facing release-notes section that is published with each tag.

### Engineering

- Package builds are tested before publication.
- PyPI publication uses GitHub Actions and Trusted Publishing.

### Verification

- Run `python -m pytest -q` before creating a release tag.
- Run `python -m build` and `python -m twine check dist/*` to validate distributions.
