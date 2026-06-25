Replaced the custom `pyphi.jsonify` JSON layer with a new `pyphi.serialize`
package built on [msgspec](https://jcristharif.com/msgspec/). It serializes
every result type through a typed, tag-discriminated schema, supports both JSON
and a compact binary msgpack format from one schema (`serialize.dumps`/`loads`/
`dump`/`load` with `format="json"` or `"msgpack"`), stores numpy arrays as their
exact `.npy` bytes (loaded with `allow_pickle=False`), and normalizes the
cause-effect structure so distinctions are stored once and relations reference
them by index — making phi-structure output dramatically smaller (the rule154
example drops from 1.3 MB to 56 KB). The per-class `to_json`/`from_json` methods
are gone. This is a format break: results saved with the old `jsonify` format
must be recomputed.
