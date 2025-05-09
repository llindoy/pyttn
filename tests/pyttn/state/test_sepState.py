from pyttn import stateStr, sepState, isKet
import pytest
import os

os.environ["OMP_NUM_THREADS"] = "1"


@pytest.mark.parametrize(
    "state, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 8, 5),
        ([0, 0, 0, 0, 0], 5, 0),
        ([], 0, 0),
    ],
)
def test_init1(state, size, nnz):
    _state = stateStr(state)
    State = sepState(state)

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == 1.0
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, 5, 0),
        ([], 3.0j, 0, 0),
    ],
)
def test_init2(state, coeff, size, nnz):
    _state = stateStr(state)
    State = sepState(coeff, state)

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 8, 5),
        ([0, 0, 0, 0, 0], 5, 0),
        ([], 0, 0),
    ],
)
def test_init3(state, size, nnz):
    _state = stateStr(state)
    State = sepState(_state)

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == 1.0
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, 5, 0),
        ([], 3.0j, 0, 0),
    ],
)
def test_init4(state, coeff, size, nnz):
    _state = stateStr(state)
    State = sepState(coeff, _state)

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, 5, 0),
        ([], 3.0j, 0, 0),
    ],
)
def test_init4(state, coeff, size, nnz):
    _state = stateStr(state)
    _State = sepState(coeff, _state)

    State = sepState(_State)

    assert State == _State
    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, 5, 0),
        ([], 3.0j, 0, 0),
    ],
)
def test_assign(state, coeff, size, nnz):
    _state = stateStr(state)
    _State = sepState(coeff, state)

    State = sepState()
    State.assign(_State)

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0),
        ([0, 0, 0, 0, 0], 2.0+1.0j),
        ([], 3.0j),
    ],
)
def test_clear(state, coeff):
    _State = sepState(coeff, state)

    State = sepState()
    State.assign(_State)

    State.clear()
    # check if the coefficient is correct
    assert State.coeff == 0
    # check it has the correct size
    assert State.size() == 0
    # and number of non-zero entries
    assert State.nnz() == 0
    # and represents the original state
    assert State.state().state() == []


@pytest.mark.parametrize(
    "state, coeff, N",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0, 2),
        ([1, 0, 2, 0, 3, 4, 0, 2], 2.0, 16),
        ([0, 0, 0, 0, 0], 3.0j, 3),
        ([0, 0, 0, 0, 0], 4.0j, 6),
        ([], 5.0+2.0j, 2),
    ],
)
def test_resize(state, coeff, N):
    State = sepState(coeff, state)
    nnz = State.nnz()
    State.resize(N)

    if N < len(state):
        # check it has the correct size
        assert State.size() == N
        # and number of non-zero entries
        assert State.nnz() == 0
        # and represents the original state
        assert State.state().state() == [0 for i in range(N)]

    else:
        # check it has the correct size
        assert State.size() == N
        # and number of non-zero entries
        assert State.nnz() == nnz
        # and represents the original state
        assert State.state().state() == state + \
            [0 for i in range(N - len(state))]


@pytest.mark.parametrize(
    "state, coeff, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, 5, 0),
        ([], 3.0j, 0, 0),
    ],
)
def test_coeff(state, coeff, size, nnz):
    _state = stateStr(state)
    State = sepState(coeff, state)
    State2 = sepState(State)

    State.coeff = 4.0

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == 4.0
    # and represents the original state
    assert State.state() == _state

    assert State != State2


@pytest.mark.parametrize(
    "state, coeff, b, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 2.0, 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, -3.0, 5, 0),
        ([], 3.0j, -2.0j, 0, 0),
    ],
)
def test_imul(state, coeff, b, size, nnz):
    _state = stateStr(state)

    State = sepState(coeff, state)
    State *= b

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff*b
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff, b, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 2.0, 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, -3.0, 5, 0),
        ([], 3.0j, -2.0j, 0, 0),
    ],
)
def test_mul(state, coeff, b, size, nnz):
    _state = stateStr(state)

    _State = sepState(coeff, state)
    State = _State * b

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff*b
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff, b, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 2.0, 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, -3.0, 5, 0),
        ([], 3.0j, -2.0j, 0, 0),
    ],
)
def test_rmul(state, coeff, b, size, nnz):
    _state = stateStr(state)

    _State = sepState(coeff, state)
    State = b*_State

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff*b
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff, b, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 2.0, 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, -3.0, 5, 0),
        ([], 3.0j, -2.0j, 0, 0),
    ],
)
def test_idiv(state, coeff, b, size, nnz):
    _state = stateStr(state)
    State = sepState(coeff, state)
    State /= b

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff/b
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state, coeff, b, size, nnz",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 2.0, 1.0, 8, 5),
        ([0, 0, 0, 0, 0], 2.0+1.0j, -3.0, 5, 0),
        ([], 3.0j, -2.0j, 0, 0),
    ],
)
def test_div(state, coeff, b, size, nnz):
    _state = stateStr(state)
    _State = sepState(coeff, state)
    State = _State/b

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert State.coeff == coeff/b
    # and represents the original state
    assert State.state() == _state


@pytest.mark.parametrize(
    "state1,c1, state2, c2",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0, [1, 0, 2, 2, 3, 4, 3, 2], 1.0j),
        ([0, 0, 0, 0, 0], 2.0, [1, 2, 3, 4, 5], 2.0j),
        ([], 3.0, [], 4.0j),
    ],
)
def test_add1(state1, c1, state2, c2):
    State = sepState(c1, state1)
    State2 = sepState(c2, state2)

    State3 = State+State2

    # check that the result of the multiplication is correct
    assert isKet(State3)


@pytest.mark.parametrize(
    "state1, state2, c2",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], [1, 0, 2, 2, 3, 4, 3, 2], 1.0j),
        ([0, 0, 0, 0, 0], [1, 2, 3, 4, 5], 2.0j),
        ([], [], 4.0j),
    ],
)
def test_add2(state1, state2, c2):
    State = stateStr(state1)
    State2 = sepState(c2, state2)

    State3 = State+State2

    # check that the result of the multiplication is correct
    assert isKet(State3)


@pytest.mark.parametrize(
    "state1, state2, c2",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], [1, 0, 2, 2, 3, 4, 3, 2], 1.0j),
        ([0, 0, 0, 0, 0], [1, 2, 3, 4, 5], 2.0j),
        ([], [], 4.0j),
    ],
)
def test_add3(state1, state2, c2):
    State = stateStr(state1)
    State2 = sepState(c2, state2)

    State3 = State2+State

    # check that the result of the multiplication is correct
    assert isKet(State3)


@pytest.mark.parametrize(
    "state1,c1, state2, c2",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0, [1, 0, 2, 2, 3, 4, 3, 2], 1.0j),
        ([0, 0, 0, 0, 0], 2.0, [1, 2, 3, 4, 5], 2.0j),
        ([], 3.0, [], 4.0j),
    ],
)
def test_sub1(state1, c1, state2, c2):
    State = sepState(c1, state1)
    State2 = sepState(c2, state2)

    State3 = State-State2

    # check that the result of the multiplication is correct
    assert isKet(State3)


@pytest.mark.parametrize(
    "state1, state2, c2",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], [1, 0, 2, 2, 3, 4, 3, 2], 1.0j),
        ([0, 0, 0, 0, 0], [1, 2, 3, 4, 5], 2.0j),
        ([], [], 4.0j),
    ],
)
def test_sub2(state1, state2, c2):
    State = stateStr(state1)
    State2 = sepState(c2, state2)

    State3 = State-State2

    # check that the result of the multiplication is correct
    assert isKet(State3)


@pytest.mark.parametrize(
    "state1, state2, c2",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], [1, 0, 2, 2, 3, 4, 3, 2], 1.0j),
        ([0, 0, 0, 0, 0], [1, 2, 3, 4, 5], 2.0j),
        ([], [], 4.0j),
    ],
)
def test_sub3(state1, state2, c2):
    State = stateStr(state1)
    State2 = sepState(c2, state2)

    State3 = State2-State

    # check that the result of the multiplication is correct
    assert isKet(State3)

