# Build and Release

> How to build, version, and release RDF-StarBase

## Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR** — Breaking API changes
- **MINOR** — New features, backward compatible
- **PATCH** — Bug fixes

Current version defined in `pyproject.toml`:
```toml
[project]
version = "0.3.0"
```

---

## Build Artifacts

### Python Package

```bash
# Install build tools
pip install build twine

# Build sdist and wheel
python -m build

# Output:
# dist/rdf_starbase-0.3.0.tar.gz
# dist/rdf_starbase-0.3.0-py3-none-any.whl
```

### Docker Image

```bash
# Build locally
docker build -t rdfstarbase:local -f deploy/docker/Dockerfile.api .

# Multi-platform build (for release)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t rxcthefirst/rdf-starbase:0.3.0 \
  -t rxcthefirst/rdf-starbase:latest \
  -f deploy/docker/Dockerfile.api \
  --push .
```

---

## Release Checklist

### Pre-Release

- [ ] All tests passing: `pytest`
- [ ] Coverage acceptable: `pytest --cov`
- [ ] No lint errors: `ruff check src/`
- [ ] Type checks pass: `mypy src/rdf_starbase`
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml`

### Release Process

1. **Create release branch**
   ```bash
   git checkout -b release/v0.3.0
   ```

2. **Update version**
   ```bash
   # Edit pyproject.toml
   version = "0.3.0"
   ```

3. **Update CHANGELOG**
   ```markdown
   ## [0.3.0] - 2026-02-05
   ### Added
   - API/engine separation
   - New authentication system
   ### Fixed
   - Import path issues
   ```

4. **Run full test suite**
   ```bash
   pytest --cov=src/rdf_starbase --cov-report=html
   ```

5. **Build and test locally**
   ```bash
   python -m build
   pip install dist/rdf_starbase-0.3.0-py3-none-any.whl
   python -c "from rdf_starbase import TripleStore; print('OK')"
   ```

6. **Create PR and merge**

7. **Tag release**
   ```bash
   git checkout main
   git pull
   git tag -a v0.3.0 -m "Release v0.3.0"
   git push origin v0.3.0
   ```

8. **Publish to PyPI**
   ```bash
   # Test PyPI first
   twine upload --repository testpypi dist/*

   # Production PyPI
   twine upload dist/*
   ```

9. **Build and push Docker**
   ```bash
   docker buildx build \
     --platform linux/amd64,linux/arm64 \
     -t rxcthefirst/rdf-starbase:0.3.0 \
     -t rxcthefirst/rdf-starbase:latest \
     --push .
   ```

10. **Create GitHub Release**
    - Go to Releases → New Release
    - Select tag v0.3.0
    - Add release notes from CHANGELOG
    - Attach wheel file

---

## CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci.yml`)

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[dev,web,query,sql]"
      - run: pytest --cov

  publish:
    needs: test
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```

---

## Hotfix Process

For urgent fixes to production:

1. Branch from latest release tag
   ```bash
   git checkout -b hotfix/0.3.1 v0.3.0
   ```

2. Make minimal fix

3. Bump patch version
   ```toml
   version = "0.3.1"
   ```

4. Test thoroughly

5. Merge to main AND release branch (if exists)

6. Tag and release as normal

---

## Rollback

If a release has critical issues:

### PyPI
```bash
# Yank the release (doesn't delete, but hides from install)
# Done via PyPI web interface
```

### Docker
```bash
# Re-tag previous version as latest
docker pull rxcthefirst/rdf-starbase:0.2.0
docker tag rxcthefirst/rdf-starbase:0.2.0 rxcthefirst/rdf-starbase:latest
docker push rxcthefirst/rdf-starbase:latest
```

### Git
```bash
# Revert the release commit
git revert <commit-hash>
git push
```
