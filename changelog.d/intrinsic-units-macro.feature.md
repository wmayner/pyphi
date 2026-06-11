Added `pyphi.macro`: the intrinsic-units macro framework of Marshall,
Findlay, Albantakis & Tononi (2024). `MacroUnit` defines a macro unit by
its constituents (micro or meso), update grain, sliding-window state
mapping, and background apportionment; `coarse_grain()` and `blackbox()`
build the paper's two mapping classes; `macro_tpms()` implements the
four-step macro TPM construction (Eqs. 26-40); and `MacroSystem` exposes
the result to the standard IIT 4.0 pipeline (`sia()`, `ces()`,
relations) exactly like a micro `System`. Identity macroing reproduces
micro results exactly, and both paper examples are reproduced at the
authors' published precision. The legacy pre-2024 `pyphi.macro` module
(`CoarseGrain`/`Blackbox`/`MacroSubsystem`) is removed, along with its
macro-only `validate` helpers and the emergence tutorial that documented
it.
