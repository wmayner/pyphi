Configuring `version: IIT_3_0` without also setting `sia_tie_resolution`
left the IIT 4.0 default (`NORMALIZED_PHI`, ...) in place, and `analyze()`
crashed with an `AttributeError` deep inside tie resolution (IIT 3.0 SIA
results have no normalized φ). Incompatible SIA tie strategies are now
rejected with a `ConfigurationError` naming the field and a fix — eagerly on
`config.override()` / `load_yaml()`, and at the analysis dispatch boundary
for configs assembled by per-field assignment.
