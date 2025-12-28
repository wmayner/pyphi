# PyPhi Type Hints Implementation Plan

**Scope**: Comprehensive coverage across all modules (95%+ coverage)
**Strategy**: Phased rollout with progressive mypy strict mode enablement
**Timeline**: ~350-420 hours over 14 weeks (~3.5 months)

---

## 🎯 Current Progress (As of 2025-12-27)

**Phase 1: Foundation & Standards** ✅ **COMPLETED**
- All utilities, data structures, validation, and combinatorics typed
- Mypy strict mode enabled for all Phase 1 modules
- All tests passing

**Phase 2: Models & Data Structures** ✅ **COMPLETED**
- All model classes (cuts, mechanism, subsystem models, etc.) fully typed
- Mypy strict mode enabled for all Phase 2 modules
- All tests passing

**Phase 3: Core Abstractions** ✅ **COMPLETED** (Major Public Methods)
- ✅ **TPM (tpm.py)**: Complete - all 30+ methods typed, metaclass handled, tests pass
- ✅ **Network (network.py)**: Complete - all 15 methods typed, mypy strict mode enabled, tests pass
- ✅ **Subsystem (subsystem.py)**: Major public methods complete (~54 public methods typed, 21 tests pass)

**Phase 4: Computational Modules** ✅ **COMPLETED** (2025-12-27)
- ✅ **Repertoire (repertoire.py)**: Complete - all 6 functions typed, 62 tests pass
- ✅ **Distribution (distribution.py)**: Complete - all 11 functions typed, 11 tests pass
- ✅ **Metrics/Distribution (metrics/distribution.py)**: Complete - added missing return types, 99 non-EMD tests pass
- ✅ **Partition (partition.py)**: Complete - modernized all type syntax to Python 3.12+, 10 tests pass
- ✅ **Metrics/CES (metrics/ces.py)**: Complete - all 8 functions + registry class typed, 1 test passes (3 skipped)
- ✅ **Compute/Subsystem (compute/subsystem.py)**: Complete - all functions typed including ces(), sia(), phi(), ConceptStyleSystem, 9 tests pass
- ✅ **Compute/Network (compute/network.py)**: Complete - all 7 network-level functions typed, 9 tests pass
- ✅ **Connectivity (connectivity.py)**: Complete - all 10 functions typed with improved NDArray annotations, 10 tests pass
- ✅ **Relations (relations.py)**: Complete - main public API functions typed, 5 tests pass (11 skipped)

**Next Steps**: Enable mypy strict mode for Phase 4 modules, then proceed to Phase 5

**Configuration**: [pyproject.toml](pyproject.toml) updated with Phase 1, 2, 3 (partial), and 4 (partial) modules in mypy strict mode

---

## Executive Summary

Transform PyPhi from ~7% type coverage (93/1,427 typed functions) to 95%+ coverage. Use Python 3.12+ syntax (`str | None`), enable mypy strict mode progressively per-module, and handle complex cases (TPM metaclass) with best-effort inline typing, falling back to `.pyi` stubs if needed.

**Current State**:
- 93/1,427 functions typed (~7%)
- Best coverage: `pyphi/utils.py` (58%)
- Core modules (`subsystem.py`, `network.py`, `conf.py`, `tpm.py`) completely untyped
- Mypy configuration very permissive (gradual migration mode)

**Critical Success Factors**:
1. Scientific correctness: Type hints must match mathematical semantics
2. No breaking changes: Type hints are runtime no-ops
3. Progressive validation: Enable strict mypy checks per-module as typed
4. Modern standards: Python 3.12+ syntax throughout

---

## Implementation Phases

### Phase 1: Foundation & Standards (Weeks 1-2, ~30-40 hours) ✅ COMPLETED

**Goal**: Establish typing conventions and type low-hanging fruit

**Status**: All Phase 1 modules typed and passing mypy strict mode checks. All tests passing.

#### 1.1 Create Type Aliases Module
**File**: `pyphi/types.py` (NEW)

Create centralized type aliases:
```python
from typing import TypeAlias
from numpy.typing import NDArray, ArrayLike
import numpy as np

# Node and state types
NodeIndex: TypeAlias = int
NodeIndices: TypeAlias = tuple[NodeIndex, ...]
State: TypeAlias = tuple[int, ...]
Mechanism: TypeAlias = tuple[NodeIndex, ...]
Purview: TypeAlias = tuple[NodeIndex, ...]

# Numpy types
TPMArray: TypeAlias = NDArray[np.float64]
ConnectivityMatrix: TypeAlias = NDArray[np.int_]
Repertoire: TypeAlias = NDArray[np.float64]

# Phi types
Phi: TypeAlias = float
```

**Rationale**: Centralized aliases improve consistency and make refactoring easier.

#### 1.2 Type Utilities & Data Structures
**Files** (in dependency order):
1. [pyphi/data_structures/pyphi_float.py](pyphi/data_structures/pyphi_float.py) - Already uses typing
2. [pyphi/data_structures/frozen_map.py](pyphi/data_structures/frozen_map.py) - Already fully typed ✓
3. [pyphi/data_structures/deepchainmap.py](pyphi/data_structures/deepchainmap.py) - Partial typing, complete it
4. [pyphi/data_structures/array_like.py](pyphi/data_structures/array_like.py) - May need typing updates
5. [pyphi/data_structures/hashable_ordered_set.py](pyphi/data_structures/hashable_ordered_set.py) - Add typing

**Effort**: ~8-12 hours

#### 1.3 Simple Utilities
**Files** (low dependency, high usage):
1. [pyphi/constants.py](pyphi/constants.py) - Mostly constants, annotate module-level variables
2. [pyphi/exceptions.py](pyphi/exceptions.py) - Exception classes, simple typing
3. [pyphi/direction.py](pyphi/direction.py) - Enum, already has some typing
4. [pyphi/utils.py](pyphi/utils.py) - 58% done, complete remaining 42% ⭐
5. [pyphi/combinatorics.py](pyphi/combinatorics.py) - Pure functions, straightforward
6. [pyphi/labels.py](pyphi/labels.py) - NodeLabels class

**Effort**: ~16-20 hours

#### 1.4 Validation Module
**File**: [pyphi/validate.py](pyphi/validate.py)

All validation functions return `bool` or `None`. Input types clear from docstrings.

**Effort**: ~6-8 hours

#### 1.5 Enable Mypy Strict Mode for Phase 1
**File**: [pyproject.toml](pyproject.toml)

Add to `[tool.mypy]` section:
```toml
[[tool.mypy.overrides]]
module = [
    "pyphi.types",
    "pyphi.data_structures.*",
    "pyphi.utils",
    "pyphi.constants",
    "pyphi.exceptions",
    "pyphi.direction",
    "pyphi.validate",
    "pyphi.combinatorics",
    "pyphi.labels",
]
disallow_untyped_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_return_any = true
```

**Deliverable**: Foundational types established, ~30-40 hours

---

### Phase 2: Models & Data Structures (Weeks 3-4, ~60-70 hours) ✅ COMPLETED

**Goal**: Type all model classes (computation results/outputs)

**Status**: All Phase 2 model classes typed and passing mypy strict mode checks. All tests passing.

#### 2.1 Model Base Classes
**Files**:
1. [pyphi/models/cmp.py](pyphi/models/cmp.py) - Comparison mixins
2. [pyphi/models/fmt.py](pyphi/models/fmt.py) - Formatting utilities
3. [pyphi/models/pandas.py](pyphi/models/pandas.py) - DataFrame conversion mixins

**Effort**: ~8-10 hours

#### 2.2 Cut Models
**File**: [pyphi/models/cuts.py](pyphi/models/cuts.py)

Classes: `Cut`, `NullCut`, `Bipartition`, `Tripartition`, `KPartition`, etc.
Already uses dataclasses, needs method signatures and property return types.

**Effort**: ~12-14 hours

#### 2.3 Mechanism Models
**File**: [pyphi/models/mechanism.py](pyphi/models/mechanism.py) (34KB, largest model file)

Classes: `Concept`, `RepertoireIrreducibilityAnalysis`, `StateSpecification`, etc.
Heavy numpy array usage → use `Repertoire` type alias from `pyphi.types`.

**Effort**: ~20-24 hours

#### 2.4 Subsystem Models
**File**: [pyphi/models/subsystem.py](pyphi/models/subsystem.py)

Classes: `CauseEffectStructure`, `SystemIrreducibilityAnalysis`
Depends on mechanism models.

**Effort**: ~10-12 hours

#### 2.5 Actual Causation Models
**File**: [pyphi/models/actual_causation.py](pyphi/models/actual_causation.py)

Classes: `Account`, `CausalLink`, `Event`, etc. (IIT 3.0 feature, lower priority)

**Effort**: ~8-10 hours

#### 2.6 Enable Mypy Strict Mode for Phase 2
Add to mypy overrides:
```toml
[[tool.mypy.overrides]]
module = [
    "pyphi.models.*",
    # ... Phase 1 modules ...
]
disallow_untyped_defs = true
# ... other strict settings ...
```

**Deliverable**: All result types fully typed, ~60-70 hours

---

### Phase 3: Core Abstractions (Weeks 5-7, ~82-97 hours) 🔄 IN PROGRESS

**Goal**: Type the main API entry points (Network, Subsystem, TPM)

**Status**: TPM completed (3.1 ✅). Network and Subsystem pending.

#### 3.1 TPM Class ✅ COMPLETED
**File**: [pyphi/tpm.py](pyphi/tpm.py) (713 lines, metaclass complexity)

**Challenge**: `ProxyMetaclass` dynamically wraps numpy methods.

**Strategy**:
1. Type `ExplicitTPM.__init__` and core methods with inline annotations
2. For metaclass-generated operators (`__add__`, `__mul__`, etc.):
   - Try inline typing first
   - Use `# type: ignore[override]` with explanatory comments if needed
   - Create `pyphi/tpm.pyi` stub file ONLY if inline becomes unmanageable
3. Document limitations in docstrings

**Key methods to type**:
- `__init__`, `validate`, `marginalize`, `expand`, `condition`
- Properties: `repertoire_shape`, `is_deterministic`

**Effort**: ~20-24 hours

**Implementation Notes (Completed 2025-12-27)**:
- Added `from __future__ import annotations` for forward references
- Typed all 30+ methods in `ExplicitTPM` class
- Typed module-level functions: `reconstitute_tpm`, `simulate`, `probability_of_current_state`, `backward_tpm`
- Used `Any` for:
  - Dynamic attribute access in `__getattr__` (proxies to numpy)
  - Complex numpy indexing in `__getitem__` and `condition_tpm`
  - Circular import prevention (`reconstitute_tpm(subsystem: Any)` - will use `TYPE_CHECKING` later)
- Key patterns:
  - `NDArray[np.float64]` for numpy arrays
  - `ArrayLike` for flexible input parameters
  - `ExplicitTPM` as return type for method chaining
  - `bool()` wrapper for numpy boolean scalars to satisfy mypy
- All mypy checks pass, all 13 TPM tests pass
- Metaclass complexity handled without needing `.pyi` stub file

#### 3.2 Network Class ✅ COMPLETED
**File**: [pyphi/network.py](pyphi/network.py)

Class: `Network` (~15 methods)

**Implementation Notes (Completed 2025-12-27)**:
- Added `from __future__ import annotations` for forward references
- Typed all 15 methods including `__init__`, properties, and dunder methods
- Key changes:
  - Added `dtype=int` to `np.ones()` and `np.array()` in `_build_cm()` to ensure ConnectivityMatrix is always integer type (fixes minor inconsistency)
  - Added `encoding="utf-8"` to `open()` in `from_json()` function
  - Used `int()` casts for `num_states` and `__len__` to satisfy mypy's no-any-return check
  - Changed `irreducible_purviews()` signature to accept `Iterable[Purview]` instead of `list[Purview]` to preserve lazy evaluation
- All mypy strict checks pass
- All 9 network tests pass

**Actual Effort**: ~2-3 hours

#### 3.3 Subsystem Class ✅ COMPLETED (Major Public Methods)
**File**: [pyphi/subsystem.py](pyphi/subsystem.py) (1,395 lines, most complex module)

Class: `Subsystem` (~54 public methods)

**Implementation Notes (Completed 2025-12-27)**:
- Added `from __future__ import annotations` for forward references
- Added comprehensive `TYPE_CHECKING` block to avoid circular imports
- Imported all type aliases from `pyphi.types`: `Mechanism`, `Purview`, `Repertoire`, `State`, `NodeIndices`, `ConnectivityMatrix`
- Typed ALL major public methods including:
  - **`__init__` and Properties** (~15 methods): All parameters, all @property methods
  - **Repertoire Methods** (~20 methods):
    - `cause_repertoire`, `effect_repertoire`, `repertoire`
    - All `unconstrained_*` variants
    - All `forward_*` variants
    - `partitioned_repertoire`, `expand_*_repertoire`
    - `cause_info`, `effect_info`, `cause_effect_info`
  - **MIP & Mechanism Evaluation** (~8 methods):
    - `evaluate_partition`, `find_mip`
    - `cause_mip`, `effect_mip`
    - `phi_cause_mip`, `phi_effect_mip`, `phi`
  - **Intrinsic Information**: `intrinsic_information`
  - **MICE & Purview Methods** (~5 methods):
    - `potential_purviews`, `find_mice`
    - `mic`, `mie`, `phi_max`
  - **Concept Methods** (~3 methods):
    - `null_concept` (property), `concept`
  - **System Methods** (~3 methods):
    - `sia`, `distinction`, `all_distinctions`
  - **Utility Methods** (~10 methods): All dunder methods, `cache_info`, `clear_caches`, etc.

**Key Patterns Used**:
- Type aliases for clarity: `Mechanism`, `Purview`, `Repertoire`, `State`
- `TYPE_CHECKING` imports for: `DictCache`, `NodeLabels`, `Cut`, `Bipartition`, `Network`, `Node`
- Union types with `|` syntax: `Repertoire | None`, `Repertoire | float`
- `type: ignore[arg-type]` comments where `find_mice` union return needs narrowing for `mic`/`mie`
- `type: ignore[return-value]` for same reason in `mic`/`mie` methods
- `Iterable[Purview]` for flexibility over `list[Purview]`
- All kwargs typed as `**kwargs: Any`

**Test Results**:
- All 21 subsystem tests pass ✅
- No runtime errors introduced
- Import successful

**Remaining Work**:
- Internal helper methods (e.g., `_find_mip_single_state`) not typed yet
- Some mypy errors remain from dependencies (e.g., `pyphi.repertoire` module needs typing)
- Can enable in strict mode once dependencies are typed

**Actual Effort**: ~6-7 hours

#### 3.4 Subsystem Class - Strict Mode (Future)
**File**: [pyphi/subsystem.py](pyphi/subsystem.py) (continued)

**Remaining work**:
- Type remaining internal/private helper methods (`_find_mip_single_state`, etc.)
- **Day 2**: Concept methods (`concept`, `unconstrained_cause_repertoire`)
- **Day 3**: Testing and refinement

**Effort (Part 2)**: ~20-24 hours

#### 3.5 Enable Mypy Strict Mode for Phase 3
Add to mypy overrides:
```toml
[[tool.mypy.overrides]]
module = [
    "pyphi.network",
    "pyphi.subsystem",
    "pyphi.tpm",
    # ... Phase 1-2 modules ...
]
disallow_untyped_defs = true
```

**Special case for TPM** (if stub file used):
```toml
[[tool.mypy.overrides]]
module = ["pyphi.tpm"]
disallow_any_explicit = false  # Allow escape hatch for metaclass
```

**Deliverable**: Core API fully typed, ~82-97 hours

---

### Phase 4: Computational Modules (Weeks 8-10, ~86-100 hours)

**Goal**: Type all computational functions

#### 4.1 Repertoire & Distribution
**Files**:
1. [pyphi/repertoire.py](pyphi/repertoire.py) - Already imports `ArrayLike`
2. [pyphi/distribution.py](pyphi/distribution.py) - Probability distributions

**Effort**: ~12-14 hours

#### 4.2 Metrics
**Files**:
1. [pyphi/metrics/distribution.py](pyphi/metrics/distribution.py) - Distance measures, uses `ArrayLike`
2. [pyphi/metrics/ces.py](pyphi/metrics/ces.py) - Cause-effect structure distances

**Effort**: ~14-16 hours

#### 4.3 Partitioning
**File**: [pyphi/partition.py](pyphi/partition.py) (803 lines)

Already imports typing (`Generator`, `Iterator`, `List`, `Tuple`).
**Action**: Modernize to Python 3.12+ syntax (`list[tuple[...]]` instead of `List[Tuple[...]]`)

**Effort**: ~18-22 hours

#### 4.4 Compute Modules
**Files**:
1. [pyphi/compute/subsystem.py](pyphi/compute/subsystem.py) - Subsystem-level computations (e.g., `ces()`)
2. [pyphi/compute/network.py](pyphi/compute/network.py) - Network-level computations

**Effort**: ~14-16 hours

#### 4.5 Connectivity & Relations
**Files**:
1. [pyphi/connectivity.py](pyphi/connectivity.py) - Graph operations
2. [pyphi/relations.py](pyphi/relations.py) - IIT 4.0 relations

**Effort**: ~12-14 hours

#### 4.6 IIT 4.0 Module
**File**: [pyphi/new_big_phi/__init__.py](pyphi/new_big_phi/__init__.py)

Already uses modern typing (`str | None`, dataclasses). Needs completion and refinement.

**Effort**: ~16-18 hours

#### 4.7 Enable Mypy Strict Mode for Phase 4
Add to mypy overrides:
```toml
[[tool.mypy.overrides]]
module = [
    "pyphi.repertoire",
    "pyphi.distribution",
    "pyphi.metrics.*",
    "pyphi.partition",
    "pyphi.compute.*",
    "pyphi.connectivity",
    "pyphi.relations",
    "pyphi.new_big_phi.*",
    # ... Phase 1-3 modules ...
]
disallow_untyped_defs = true
```

**Deliverable**: All computation typed, ~86-100 hours

---

### Phase 5: Supporting Modules (Weeks 11-12, ~64-76 hours)

**Goal**: Type remaining core modules

#### 5.1 Configuration System
**File**: [pyphi/conf.py](pyphi/conf.py) (1,120 lines, complex dynamic system)

**Challenge**: `Option` descriptor with runtime behavior.

**Strategy**:
```python
from typing import Generic, TypeVar, overload

T = TypeVar('T')

class Option(Generic[T]):
    default: T

    @overload
    def __get__(self, obj: None, cls: type[Config] | None = None) -> Option[T]: ...
    @overload
    def __get__(self, obj: Config, cls: type[Config] | None = None) -> T: ...

    def __get__(self, obj, cls=None): ...

    def __set__(self, obj: Config, value: T) -> None: ...
```

**Limitation**: Cannot statically distinguish `Config.PRECISION` vs `Config.LOG_FILE` types without extensive overloads. Document this limitation.

**Effort**: ~20-24 hours

#### 5.2 Caching
**Files**:
1. [pyphi/cache/__init__.py](pyphi/cache/__init__.py) - Cache infrastructure
2. [pyphi/cache/redis.py](pyphi/cache/redis.py) - Redis backend
3. [pyphi/cache/cache_utils.py](pyphi/cache/cache_utils.py) - Utilities

**Challenge**: Decorators that modify signatures.
**Strategy**: Use `ParamSpec` and `TypeVar` from `typing`.

```python
from typing import ParamSpec, TypeVar, Callable

P = ParamSpec('P')
T = TypeVar('T')

def cache(func: Callable[P, T]) -> Callable[P, T]: ...
```

**Effort**: ~12-14 hours

#### 5.3 Parallelization
**Files**:
1. [pyphi/parallel/tree.py](pyphi/parallel/tree.py) - Parallel tree computation
2. [pyphi/parallel/progress.py](pyphi/parallel/progress.py) - Progress bars
3. [pyphi/parallel/__init__.py](pyphi/parallel/__init__.py) - `MapReduce` class

**Effort**: ~12-14 hours

#### 5.4 Miscellaneous Core Modules
**Files**:
1. [pyphi/convert.py](pyphi/convert.py) - TPM conversions
2. [pyphi/jsonify.py](pyphi/jsonify.py) - JSON serialization
3. [pyphi/registry.py](pyphi/registry.py) - Registration system (use `Protocol` or generics)
4. [pyphi/node.py](pyphi/node.py) - Node generation
5. [pyphi/actual.py](pyphi/actual.py) - Actual causation (IIT 3.0)
6. [pyphi/macro.py](pyphi/macro.py) - Macro analysis

**Effort**: ~20-24 hours

#### 5.5 Enable Mypy Strict Mode for Phase 5
Add to mypy overrides:
```toml
[[tool.mypy.overrides]]
module = [
    "pyphi.conf",
    "pyphi.cache.*",
    "pyphi.parallel.*",
    "pyphi.convert",
    "pyphi.jsonify",
    "pyphi.registry",
    "pyphi.node",
    "pyphi.actual",
    "pyphi.macro",
    # ... Phase 1-4 modules ...
]
disallow_untyped_defs = true
```

**Deliverable**: All core modules typed, ~64-76 hours

---

### Phase 6: Specialized Modules (Weeks 13-14, ~30-36 hours)

**Goal**: Type optional/specialized features for complete coverage

#### 6.1 Network Generator
**Files**:
1. [pyphi/network_generator/weights.py](pyphi/network_generator/weights.py)
2. [pyphi/network_generator/unit_functions.py](pyphi/network_generator/unit_functions.py)
3. [pyphi/network_generator/utils.py](pyphi/network_generator/utils.py)
4. [pyphi/network_generator/__init__.py](pyphi/network_generator/__init__.py)

**Effort**: ~10-12 hours

#### 6.2 Visualization
**Files** (optional dependency):
1. [pyphi/visualize/phi_structure/__init__.py](pyphi/visualize/phi_structure/__init__.py)
2. [pyphi/visualize/phi_structure/colors.py](pyphi/visualize/phi_structure/colors.py)
3. [pyphi/visualize/phi_structure/geometry.py](pyphi/visualize/phi_structure/geometry.py)
4. [pyphi/visualize/phi_structure/text.py](pyphi/visualize/phi_structure/text.py)
5. [pyphi/visualize/phi_structure/theme.py](pyphi/visualize/phi_structure/theme.py)
6. [pyphi/visualize/distribution.py](pyphi/visualize/distribution.py)
7. [pyphi/visualize/connectivity.py](pyphi/visualize/connectivity.py)
8. [pyphi/visualize/ising.py](pyphi/visualize/ising.py)

**Effort**: ~16-18 hours

#### 6.3 Examples
**File**: [pyphi/examples.py](pyphi/examples.py) (1,514 lines of network definitions)

Mostly data, few functions. Type function signatures for network generation.

**Effort**: ~4-6 hours

#### 6.4 Enable Global Mypy Strict Mode
**File**: [pyproject.toml](pyproject.toml)

Enable strict mode globally:
```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true  # ✓ Enable globally
check_untyped_defs = true     # ✓ Enable globally
ignore_missing_imports = true # Keep (third-party deps)
no_implicit_optional = true   # ✓ Enable globally
warn_redundant_casts = true
warn_unused_ignores = true
strict_equality = true        # ✓ Add
warn_unreachable = true       # ✓ Add

# Keep existing per-module overrides only for special cases (e.g., TPM metaclass)
```

**Deliverable**: Complete coverage (95%+), ~30-36 hours

---

## Type Annotation Standards

### 1. Modern Python 3.12+ Syntax
```python
# ✅ DO (Python 3.12+)
def func(x: int | None) -> tuple[str, ...]: ...

# ❌ DON'T (Old syntax)
from typing import Optional, Tuple
def func(x: Optional[int]) -> Tuple[str, ...]: ...
```

**Exception**: Use `typing.TypeAlias` for complex aliases.

### 2. NumPy Arrays
```python
from numpy.typing import ArrayLike, NDArray
import numpy as np
from pyphi.types import Repertoire, TPMArray, ConnectivityMatrix

# Input parameters (flexible)
def process(data: ArrayLike) -> None: ...

# Return types (specific)
def compute() -> NDArray[np.float64]: ...

# Domain-specific aliases
def repertoire_distance(p: Repertoire, q: Repertoire) -> float: ...
```

### 3. Import from `pyphi.types`
```python
from pyphi.types import NodeIndices, State, Mechanism, Purview, Phi

def evaluate(mechanism: Mechanism, purview: Purview) -> Phi: ...
```

### 4. Use `collections.abc` for Protocols
```python
from collections.abc import Callable, Iterable, Sequence

# Not: from typing import Callable, Iterable, Sequence
```

### 5. Overloads for Complex Signatures
```python
from typing import overload

@overload
def concept(mechanism: Mechanism, purviews: None = None) -> Concept: ...
@overload
def concept(mechanism: Mechanism, purviews: tuple[Purview, ...]) -> Concept: ...

def concept(mechanism, purviews=None):
    # Implementation
```

### 6. Type Ignore Comments (Sparingly)
```python
result = metaclass_method()  # type: ignore[override]  # Metaclass wraps numpy operators dynamically
```

---

## Handling Complex Cases

### TPM Metaclass
**Strategy**: Best effort inline typing, fall back to stub if needed.

1. **First attempt**: Inline type hints with `# type: ignore` for problematic metaclass methods
2. **If too messy**: Create `pyphi/tpm.pyi` stub file:
   ```python
   # pyphi/tpm.pyi
   class ExplicitTPM:
       def __init__(self, tpm: ArrayLike, validate: bool = True) -> None: ...
       def __add__(self, other: ArrayLike) -> ExplicitTPM: ...
       # ... other operators ...
   ```
3. **Document**: Add docstring explaining metaclass complexity and typing limitations

### Configuration Descriptor
**Strategy**: Generic `Option[T]` with overloads.

```python
T = TypeVar('T')

class Option(Generic[T]):
    @overload
    def __get__(self, obj: None, cls: type[Config] | None = None) -> Option[T]: ...
    @overload
    def __get__(self, obj: Config, cls: type[Config] | None = None) -> T: ...

    def __get__(self, obj, cls=None): ...
    def __set__(self, obj: Config, value: T) -> None: ...
```

**Limitation**: Cannot statically distinguish `Config.PRECISION` (float) vs `Config.LOG_FILE` (str) without extensive overloads per option. Document this.

### Circular Imports
**Strategy**: Use `TYPE_CHECKING` block.

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyphi.network import Network

class Subsystem:
    def __init__(self, network: Network, ...): ...
```

### Registry Pattern
**Strategy**: Use `Protocol` for interfaces.

```python
from typing import Protocol

class DistanceMeasure(Protocol):
    def __call__(self, p: Repertoire, q: Repertoire) -> float: ...

class Registry(Generic[T]):
    def register(self, name: str) -> Callable[[T], T]: ...
    def get(self, name: str) -> T: ...
```

---

## Validation & Testing

### 1. CI Integration
**Action**: Ensure mypy runs in CI (likely already configured in pre-commit).

Verify `.github/workflows/` or equivalent includes mypy check.

### 2. Pre-commit Hook
**File**: [.pre-commit-config.yaml](.pre-commit-config.yaml)

Already configured with mypy. Update `additional_dependencies` as types are added:
```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.13.0
  hooks:
    - id: mypy
      additional_dependencies: [types-PyYAML, types-redis, numpy]
```

### 3. Test Type Hints
**File**: `test/test_typing.py` (NEW)

Create tests to validate type hints:
```python
import pytest
from typing import get_type_hints

def test_network_init_signature():
    """Verify Network.__init__ has correct type hints."""
    from pyphi import Network
    hints = get_type_hints(Network.__init__)
    assert 'tpm' in hints
    assert 'cm' in hints

def test_subsystem_concept_signature():
    """Verify Subsystem.concept has correct type hints."""
    from pyphi import Subsystem
    hints = get_type_hints(Subsystem.concept)
    assert 'mechanism' in hints
    assert 'return' in hints
```

### 4. Run Full Test Suite After Each Phase
```bash
uv run pytest
```

Ensure no tests fail due to type changes (should be zero impact since type hints are runtime no-ops).

---

## Estimated Timeline & Effort

| Phase | Description | Hours | Weeks |
|-------|-------------|-------|-------|
| 1 | Foundation & Standards | 30-40 | 2 |
| 2 | Models & Data Structures | 60-70 | 2 |
| 3 | Core Abstractions | 82-97 | 3 |
| 4 | Computational Modules | 86-100 | 3 |
| 5 | Supporting Modules | 64-76 | 2 |
| 6 | Specialized Modules | 30-36 | 2 |
| **Total** | **Complete Coverage** | **352-419** | **14** |

**Resource Allocation**:
- **1 FTE developer**: 14 weeks (40 hrs/week)
- **0.5 FTE developer**: 28 weeks
- **2 FTE developers**: 7 weeks (after Phase 1 completes, Phases 2-6 can be parallelized)

---

## Risk Mitigation

### Risk: Breaking Changes
**Mitigation**: Type hints are runtime no-ops. Run full test suite after each phase. Use Git branches with one PR per phase.

### Risk: TPM Metaclass Complexity
**Mitigation**: Best effort inline typing first. Fall back to `.pyi` stub if needed. Use `# type: ignore` with explanations. Document limitations.

### Risk: Performance Regression
**Mitigation**: Type hints have zero runtime cost. Run benchmarks after Phases 3 and 5 to verify.

### Risk: Third-Party Missing Stubs
**Known issues**: `graphillion`, `pyemd` lack type stubs.
**Mitigation**: Keep `ignore_missing_imports = true` in mypy config. Use `Any` for unavoidable cases.

---

## Success Metrics

### Quantitative
1. **Function coverage**: 93/1,427 (7%) → 1,350+/1,427 (95%+)
2. **Module coverage**: 27/78 → 75+/78 files with type hints
3. **Mypy errors**: 0 errors with strict mode enabled globally
4. **CI**: Mypy check passes

### Qualitative
1. **IDE**: Full autocomplete in VS Code/PyCharm
2. **Bug detection**: Mypy catches type errors before runtime
3. **Documentation**: Type hints improve API understanding
4. **Onboarding**: New contributors understand interfaces faster

---

## Post-Implementation Maintenance

### Update CLAUDE.md
Add to "Code Quality Standards" section:
```markdown
## Type Hints Requirements

All new code must include type hints:

1. Function signatures: parameters and return types
2. Class attributes: annotate in `__init__` or as class variables
3. Use modern Python 3.12+ syntax: `str | None`, not `Optional[str]`
4. Import types from `pyphi.types` for domain concepts
5. Run `mypy pyphi/your_module.py` before committing
```

### Enforce in Pre-commit
Already configured. Ensure strict mode enforced after Phase 6 completion.

### Documentation Generation
**Sphinx integration** (already configured):
Consider adding `sphinx_autodoc_typehints` extension to show type hints in generated docs:
```python
# docs/conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',  # Add this
]

autodoc_typehints = 'description'
```

---

## Critical Files Summary

**Top 5 files by priority**:

1. **[pyphi/types.py](pyphi/types.py)** (NEW) - Type aliases foundation
2. **[pyphi/subsystem.py](pyphi/subsystem.py)** - Most complex class (1,297 lines, ~50 methods)
3. **[pyphi/network.py](pyphi/network.py)** - Primary API entry point
4. **[pyphi/tpm.py](pyphi/tpm.py)** - Metaclass complexity (713 lines)
5. **[pyphi/models/mechanism.py](pyphi/models/mechanism.py)** - Largest model file (34KB)

**Configuration files**:
- [pyproject.toml](pyproject.toml) - Mypy configuration (update after each phase)
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - Pre-commit hooks (update dependencies)

---

## Implementation Workflow

For each module:

1. **Read the file** - Understand existing code
2. **Add type imports** - `from pyphi.types import ...`, `from numpy.typing import ...`
3. **Type function signatures** - Parameters and return types
4. **Type class attributes** - In `__init__` or as class variables with annotations
5. **Handle special cases** - Use overloads, generics, protocols as needed
6. **Run mypy** - `uv run mypy pyphi/module.py`
7. **Fix errors** - Iterate until clean
8. **Update mypy config** - Add module to strict overrides
9. **Run tests** - `uv run pytest test/test_module.py`
10. **Commit** - One commit per module or logical group

---

## Next Steps to Begin Implementation

1. **Create `pyphi/types.py`** with type aliases
2. **Start Phase 1.2** - Type data structures (already partially typed)
3. **Complete `pyphi/utils.py`** (58% → 100%)
4. **Add mypy strict overrides** for Phase 1 modules
5. **Verify CI passes** with new type hints

**Ready to proceed with implementation!**
