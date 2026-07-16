Serialized documents now carry node labels once, on the document envelope,
instead of duplicating them into every nested object; nested objects
inherit the frame on decode, and objects whose labels differ from the
frame keep per-object labels. `pyphi.serialize.loads()` accepts a
`node_labels` argument that replaces the stored frame on load. Documents
written by earlier versions load unchanged.
