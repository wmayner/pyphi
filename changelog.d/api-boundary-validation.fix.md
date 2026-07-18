Several construction-time validation gaps are closed: `Transition` validates
its states eagerly (matching `TransitionSystem`); numpy integer states are
accepted whenever integer indices are (e.g. states produced by
`dynamics.simulate`); node-index arguments are range-checked with a clear
error instead of failing deep in the TPM backend; a wrong-shaped TPM error
names the accepted input forms; `System.from_substrate` forwards keyword
arguments (previously `background_conditioning` passed there was silently
dropped, changing cause repertoires); and `basic_substrate(cm=...)` honors the
passed connectivity matrix instead of silently discarding it.
