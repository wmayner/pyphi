import subprocess
import sys


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
