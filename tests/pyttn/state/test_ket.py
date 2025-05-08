from pyttn import stateStr, sepState, ket
import pytest
import os
from random import randrange, uniform

os.environ["OMP_NUM_THREADS"] = "1"


def test_init():
    State = ket()

    # check it has the correct size
    assert State.nterms() == 0
    # and number of non-zero entries
    assert not State.contains(stateStr([1, 0, 2, 0]))


def generate_state(N):
    return [randrange(0, 10) for i in range(N)]


def generate_states(N, nterms):
    return [generate_state(N) for i in range(nterms)]


def generate_coeffs(nterms):
    return [uniform(0, 1)+1.0j*uniform(0, 1) for i in range(nterms)]


def increment_state(state, ninc):
    return [x+ninc for x in state]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 1),
        (2, 0),
    ],
)
def test_build_ket(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    # check it has the correct size
    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 1),
        (2, 0),
    ],
)
def test_assign(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    _State = ket()
    for i in range(nterms):
        _State += coeffs[i]*stateStr(states[i])

    State = ket()
    State.assign(_State)

    # check it has the correct size
    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 1),
        (2, 0),
    ],
)
def test_clear(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    State.clear()

    # check it has the correct size
    assert State.nterms() == 0

    for i in range(nterms):
        assert not State.contains(stateStr(states[i]))


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 1),
        (2, 0),
    ],
)
def test_contains(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    _State = ket()
    for i in range(nterms):
        _State += coeffs[i]*stateStr(states[i])

    State = ket()
    State.assign(_State)

    # check it has the correct size
    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert not State.contains(stateStr(increment_state(states[i], 10)))


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 1),
        (2, 0),
    ],
)
def test_setitem(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    _State = ket()
    for i in range(nterms):
        _State += coeffs[i]*stateStr(states[i])

    State = ket()
    State.assign(_State)

    # check it has the correct size
    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    for i in range(nterms):
        State[stateStr(states[i])] = i+1.0j

    for i in range(nterms):
        assert State[stateStr(states[i])] == i+1.0j


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 1),
        (2, 0),
    ],
)
def test_insert1(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    _State = ket()
    for i in range(nterms):
        _State += coeffs[i]*stateStr(states[i])

    State = ket()
    State.assign(_State)

    # check it has the correct size
    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    # now insert all the same states with new coefficients
    for i in range(nterms):
        State.insert(sepState(i+1.0j, states[i]))

    assert State.nterms() == nterms
    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == (coeffs[i]+i+1.0j)


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 1),
        (2, 0),
    ],
)
def test_insert2(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    _State = ket()
    for i in range(nterms):
        _State += coeffs[i]*stateStr(states[i])

    State = ket()
    State.assign(_State)

    # check it has the correct size
    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    # now insert all the same states with new coefficients
    for i in range(nterms):
        State.insert(
            sepState(i+1.0j, stateStr(increment_state(states[i], 10))))

    assert State.nterms() == 2*nterms
    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == (coeffs[i])
        assert State.contains(stateStr(increment_state(states[i], 10)))
        assert State[stateStr(increment_state(states[i], 10))] == (i+1.0j)


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_prune_zeros(N, nterms):
    coeffs = [(i+1) % 2 for i in range(nterms)]
    states = generate_states(N, nterms)

    _State = ket()
    for i in range(nterms):
        _State += coeffs[i]*stateStr(states[i])

    State = ket()
    State.assign(_State)

    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    State.prune_zeros()
    assert State.nterms() == nterms//2

    for i in range(nterms//2):
        assert State.contains(stateStr(states[i*2]))
        assert State[stateStr(states[i*2])] == 1

        assert not State.contains(stateStr(states[i*2+1]))
        with pytest.raises(RuntimeError):
            State[stateStr(states[i*2+1])]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_imul(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    State *= 2.0j

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]*2.0j


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_idiv(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    State /= 2.0j

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]/2.0j


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_iadd1(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    states2 = generate_states(N, nterms)
    states2 = [i for i in states2 if i not in states]

    for i in range(len(states2)):
        State += stateStr(states2[i])

    assert State.nterms() == nterms + len(states2)

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    for i in range(len(states2)):
        assert State.contains(stateStr(states2[i]))
        assert State[stateStr(states2[i])] == 1


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_iadd2(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    for i in range(len(states)):
        State += stateStr(states[i])

    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]+1


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_iadd3(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    states2 = generate_states(N, nterms)
    states2 = [i for i in states2 if i not in states]
    coeffs2 = generate_coeffs(len(states2))

    for i in range(len(states2)):
        State += coeffs2[i]*stateStr(states2[i])

    assert State.nterms() == nterms + len(states2)

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    for i in range(len(states2)):
        assert State.contains(stateStr(states2[i]))
        assert State[stateStr(states2[i])] == coeffs2[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_iadd4(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    coeffs2 = generate_coeffs(nterms)

    for i in range(len(states)):
        State += coeffs2[i]*stateStr(states[i])

    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]+coeffs2[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_iadd5(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    states2 = generate_states(N, nterms)
    states2 = [i for i in states2 if i not in states]
    coeffs2 = generate_coeffs(len(states2))

    State2 = ket()
    for i in range(len(states2)):
        State2 += coeffs2[i]*stateStr(states2[i])

    State += State2
    assert State.nterms() == nterms + len(states2)

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    for i in range(len(states2)):
        assert State.contains(stateStr(states2[i]))
        assert State[stateStr(states2[i])] == coeffs2[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_iadd6(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    coeffs2 = generate_coeffs(nterms)

    State2 = ket()
    for i in range(len(states)):
        State2 += coeffs2[i]*stateStr(states[i])

    State += State2
    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]+coeffs2[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_isub1(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    states2 = generate_states(N, nterms)
    states2 = [i for i in states2 if i not in states]

    for i in range(len(states2)):
        State -= stateStr(states2[i])

    assert State.nterms() == nterms + len(states2)

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    for i in range(len(states2)):
        assert State.contains(stateStr(states2[i]))
        assert State[stateStr(states2[i])] == -1


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_isub2(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    for i in range(len(states)):
        State -= stateStr(states[i])

    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]-1


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_isub3(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    states2 = generate_states(N, nterms)
    states2 = [i for i in states2 if i not in states]
    coeffs2 = generate_coeffs(len(states2))

    for i in range(len(states2)):
        State -= coeffs2[i]*stateStr(states2[i])

    assert State.nterms() == nterms + len(states2)

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    for i in range(len(states2)):
        assert State.contains(stateStr(states2[i]))
        assert State[stateStr(states2[i])] == -coeffs2[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_isub4(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    coeffs2 = generate_coeffs(nterms)

    for i in range(len(states)):
        State -= coeffs2[i]*stateStr(states[i])

    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]-coeffs2[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_isub5(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    states2 = generate_states(N, nterms)
    states2 = [i for i in states2 if i not in states]
    coeffs2 = generate_coeffs(len(states2))

    State2 = ket()
    for i in range(len(states2)):
        State2 += coeffs2[i]*stateStr(states2[i])

    State -= State2
    assert State.nterms() == nterms + len(states2)

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]

    for i in range(len(states2)):
        assert State.contains(stateStr(states2[i]))
        assert State[stateStr(states2[i])] == -coeffs2[i]


@pytest.mark.parametrize(
    "N, nterms",
    [
        (10, 1000),
        (5, 2),
        (2, 0),
    ],
)
def test_isub6(N, nterms):
    coeffs = generate_coeffs(nterms)
    states = generate_states(N, nterms)

    State = ket()
    for i in range(nterms):
        State += coeffs[i]*stateStr(states[i])

    coeffs2 = generate_coeffs(nterms)

    State2 = ket()
    for i in range(len(states)):
        State2 += coeffs2[i]*stateStr(states[i])

    State -= State2
    assert State.nterms() == nterms

    for i in range(nterms):
        assert State.contains(stateStr(states[i]))
        assert State[stateStr(states[i])] == coeffs[i]-coeffs2[i]


"""
py::class_<_ket>(m, ("ket_" + label).c_str())

         .def("reserve", &_ket::reserve, "For details see :meth:`pyttn.ket_dtype.reserve`")
         .def("__str__", [](const _ket &o)
              {std::ostringstream oss; oss << o; return oss.str(); })

         .def("__iter__", [](_ket &s)
              { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())

         .def("__add__", [](_ket &a, _sepState &b)
              { return a + b; })
         .def("__add__", [](_ket &a, stateStr &b)
              { return a + b; })
         .def("__add__", [](_ket &a, ttns::sepState<real_type> &b)
              { return a + b; })
         .def("__add__", [](_ket &a, ttns::sepState<complex_type> &b)
              { return a + b; })
         .def("__add__", [](_ket &a, _ket &b)
              { return a + b; })
         .def("__add__", [](_ket &a, ttns::ket<real_type> &b)
              { return a + b; })
         .def("__add__", [](_ket &a, ttns::ket<complex_type> &b)
              { return a + b; })

         .def("__sub__", [](_ket &a, _sepState &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, stateStr &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, ttns::sepState<real_type> &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, ttns::sepState<complex_type> &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, _ket &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, ttns::ket<real_type> &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, ttns::ket<complex_type> &b)
              { return a - b; });
"""
