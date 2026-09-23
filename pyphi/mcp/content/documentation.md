# Reading the PyPhi documentation

The reference topics on this server are condensed. The full documentation site
is at https://pyphi.readthedocs.io/en/stable/ (replace `stable` with `latest`
for the development version, or with a tag such as `v2.0.0`). It publishes
three files for programmatic reading; all paths below are relative to that
site root.

- `llms.txt` lists every narrative page with a one-line summary and a link to
  its markdown source. Fetch it first to find the page you need.
- `llms-full.txt` is the whole narrative site as one markdown file (roughly
  370 KB). The generated API pages are left out of it.
- `_sources/<path>.md.txt` is the markdown source of any page, where `<path>`
  is the page's path without `.html`; the how-to on building substrates, for
  example, is `_sources/howto/build-substrate.md.txt`.
- `objects.inv` is the Sphinx intersphinx inventory: it maps every documented
  module, class, function, and method to its page and anchor. Read it with
  `sphinx.ext.intersphinx.inspect_main` or the `sphobjinv` package.
- `sitemap.xml` lists every page URL, including the API pages.

## What the sections cover

- **Getting started**: installation and a first computation.
- **Tutorials**: worked, executed examples: a complete IIT analysis, macro
  units and grains, recursive exclusion, actual causation, causal reductionism.
- **How-to guides**: one task each: build a substrate, read a result, estimate
  cost, configure, run in parallel, sweep, run a cluster campaign, cache,
  export, query relations, search across grains, visualize, use this MCP
  server, and reproduce results from earlier versions of IIT.
- **Theory**: how IIT's mathematics maps onto PyPhi's objects: substrate and
  system, conditional independence, the intrinsic-information requirement,
  distinctions and relations, the Φ-structure, macro units, and computational
  complexity.
- **Reference**: the curated API page, the executed gallery of example
  networks, the configuration options, the glossary, and the conventions for
  transition probability matrices and states.
- **Migration**: moving to PyPhi 2.0 from 1.x and from related tools.

The condensed topics on this server (`get_iit_reference(topic)`) are the place
to start; go to the site when a task needs a page's full worked example or an
API detail the topics do not carry.
