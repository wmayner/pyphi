import subprocess
import sys
import threading
import time


def test_registration_is_deferred_to_first_use():
    # Importing the serializer must not register any types (registration imports
    # the domain tree); the registries stay empty until the first encode/decode.
    # A subprocess gives a clean module state independent of the test session.
    code = (
        "import pyphi.serialize.convert as c; "
        "assert not c._REGISTERED, 'registration ran at import time'; "
        "assert not c._ENCODERS and not c._DECODERS, 'registries populated at import'; "
        "from pyphi import examples, serialize; "
        "serialize.dumps(examples.basic_system().sia()); "
        "assert c._REGISTERED, 'first encode did not trigger registration'; "
        "assert c._ENCODERS and c._DECODERS, 'registries empty after first encode'; "
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_round_trip_still_works_after_lazy_registration():
    from pyphi import examples
    from pyphi import serialize

    sia = examples.basic_system().sia()
    assert serialize.loads(serialize.dumps(sia)) == sia


def test_concurrent_first_use_never_sees_partial_registry(monkeypatch):
    """A thread encoding during another thread's first-use registration must
    see either no registry (and wait) or the complete one — never a partial
    registry ("No serializer registered for Direction")."""
    from pyphi.direction import Direction
    from pyphi.serialize import convert

    # Establish the fully-registered baseline, then reset to the
    # pre-first-use state. Registration is idempotent, so the module is
    # left fully registered again when the test ends.
    convert._ensure_registered()
    monkeypatch.setattr(convert, "_REGISTERED", False)
    convert._ENCODERS.clear()
    convert._DECODERS.clear()

    entered = threading.Event()
    orig_first = convert._register_direction

    def slow_first():
        entered.set()
        # Hold registration open while the observer thread encodes.
        time.sleep(0.2)
        orig_first()

    monkeypatch.setattr(convert, "_register_direction", slow_first)
    errors = []

    def observe():
        entered.wait(timeout=10)
        try:
            convert.to_schema(Direction.CAUSE)
        except Exception as exc:
            errors.append(exc)

    registrar = threading.Thread(target=convert._ensure_registered)
    observer = threading.Thread(target=observe)
    registrar.start()
    observer.start()
    registrar.join(timeout=30)
    observer.join(timeout=30)
    assert not errors, errors
    assert convert._REGISTERED
    assert convert._ENCODERS and convert._DECODERS
