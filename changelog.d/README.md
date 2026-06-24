# Changelog Fragments

This directory contains "news fragments" for [Towncrier](https://towncrier.readthedocs.io/).

## Creating a Fragment

When making a user-facing change, create a fragment file:

```bash
# Using towncrier create (recommended)
uv run towncrier create <name>.<type>.md

# Or manually create a file
echo "Description of change" > changelog.d/<name>.<type>.md
```

Where:
- `<name>` is either a GitHub issue number (e.g., `123`) or a descriptive name (e.g., `fix-cache-bug`)
- `<type>` is one of the fragment types below

## Fragment Types

| Type | Description |
|------|-------------|
| `feature` | New API additions |
| `change` | Changes to existing API |
| `config` | Configuration changes |
| `optimization` | Performance improvements |
| `fix` | Bug fixes |
| `doc` | Documentation updates |
| `refactor` | Code refactoring |
| `misc` | Other changes (not shown in changelog) |

## Examples

```bash
# New feature
echo "Added \`pyphi.tpm.simulate()\` function" > changelog.d/simulate.feature.md

# Bug fix with issue number
echo "Fixed repertoire calculation for edge cases" > changelog.d/456.fix.md

# Configuration change
echo "Added \`VALIDATE_JSON_VERSION\` config option" > changelog.d/json-validation.config.md
```

## Building the Changelog

At release time, fragments are compiled into CHANGELOG.md:

```bash
uv run towncrier build --version X.Y.Z
```

Use `--draft` to preview without making changes:

```bash
uv run towncrier build --draft
```
