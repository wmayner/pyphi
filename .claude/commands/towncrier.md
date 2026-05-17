---
description: Create towncrier news fragments for repository changes
argument-hint: [focus]
allowed-tools: Read, Glob, Grep, Write, Bash(git status*), Bash(git diff*), Bash(git log*), Bash(git show*), Bash(towncrier *)
---

# Create Towncrier News Fragments

Create news fragment files for towncrier based on changes in the current repository.

## Arguments
- `$ARGUMENTS` - Optional: Specify the area or scope of changes to focus on (e.g., "api", "cli", "authentication", "database"). If not provided, analyze all changes.

## Instructions

1. **Find the news fragments directory**: Look for common towncrier fragment directories:
   - `newsfragments/`
   - `news/`
   - `changelog.d/`
   - `changes/`
   - Check `pyproject.toml` or `towncrier.toml` for the configured directory

2. **Determine fragment types**: Check the towncrier configuration for valid fragment types. Common types include:
   - `feature` - New features
   - `bugfix` - Bug fixes
   - `doc` - Documentation changes
   - `removal` - Deprecations or removals
   - `misc` - Miscellaneous changes
   - `breaking` - Breaking changes

3. **Analyze repository changes**:
   - Check `git status` for staged and unstaged changes
   - Use `git diff` to understand what changed
   - If no uncommitted changes, look at recent commits with `git log`
   - If `$ARGUMENTS` specifies an area, filter changes to focus on that area

4. **Create news fragments**:
   - Fragment filename format is typically: `<issue_number>.<type>.rst` or `<issue_number>.<type>.md`
   - If no issue number is available, use a descriptive slug or `+<unique_id>`
   - Write concise, user-facing descriptions of each change
   - Each fragment should describe ONE change
   - Use imperative mood (e.g., "Add feature X" not "Added feature X")

5. **Fragment content guidelines**:
   - Keep descriptions brief but informative
   - Focus on what changed from a user's perspective
   - Include any breaking change warnings
   - Reference related issues/PRs if known

## Example Fragment Content

For a feature:
```
Add support for custom authentication providers
```

For a bugfix:
```
Fix crash when processing empty configuration files
```

For a breaking change:
```
Remove deprecated ``legacy_mode`` configuration option. Use ``compatibility_mode`` instead.
```

## Focus Area Filter

If `$ARGUMENTS` is provided (value: "$ARGUMENTS"), focus your analysis on changes related to that area. For example:
- If "api" is specified, focus on API-related changes
- If "cli" is specified, focus on command-line interface changes
- If empty or not specified, analyze all significant changes

Now analyze the repository and create appropriate news fragments.
