from pyttn import stateStr, isSepState, isKet
import pytest
import os

os.environ["OMP_NUM_THREADS"] = "1"


@pytest.mark.parametrize(
    "state, size, nnz, outstring",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 8, 5, "1 0 2 0 3 4 0 2"),
        ([0, 0, 0, 0, 0], 5, 0, "0 0 0 0 0"),
        ([], 0, 0, ""),
    ],
)
def test_init(state, size, nnz, outstring):
    State = stateStr(state)

    # check it has the correct size
    assert State.size() == size
    # and number of non-zero entries
    assert State.nnz() == nnz
    # and the correct string representation
    assert str(State) == outstring
    # and represents the original state
    assert State.state() == state


@pytest.mark.parametrize(
    "state, size, nnz, outstring",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 8, 5, "1 0 2 0 3 4 0 2"),
        ([0, 0, 0, 0, 0], 5, 0, "0 0 0 0 0"),
        ([], 0, 0, ""),
    ],
)
# now we test the copy constructor
def test_copy_init(state, size, nnz, outstring):
    State = stateStr(state)

    State2 = stateStr(State)
    # check it has the correct size
    assert State2.size() == size
    # and number of non-zero entries
    assert State2.nnz() == nnz
    # and the correct string representation
    assert str(State2) == outstring
    # and represents the original state
    assert State2.state() == state


@pytest.mark.parametrize(
    "state, size, nnz, outstring",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 8, 5, "1 0 2 0 3 4 0 2"),
        ([0, 0, 0, 0, 0], 5, 0, "0 0 0 0 0"),
        ([], 0, 0, ""),
    ],
)
# now we test the copy assignment
def test_assign(state, size, nnz, outstring):
    State = stateStr(state)

    State2 = stateStr()
    State2.assign(State)

    # check it has the correct size
    assert State2.size() == size
    # and number of non-zero entries
    assert State2.nnz() == nnz
    # and the correct string representation
    assert str(State2) == outstring
    # and represents the original state
    assert State2.state() == state


@pytest.mark.parametrize(
    "state",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2]),
        ([0, 0, 0, 0, 0]),
        ([]),
    ],
)
def test_clear(state):
    State = stateStr(state)

    State.clear()
    # check it has the correct size
    assert State.size() == 0
    # and number of non-zero entries
    assert State.nnz() == 0
    # and the correct string representation
    assert str(State) == ""
    # and represents the original state
    assert State.state() == []


@pytest.mark.parametrize(
    "state, N",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 2),
        ([1, 0, 2, 0, 3, 4, 0, 2], 16),
        ([0, 0, 0, 0, 0], 3),
        ([0, 0, 0, 0, 0], 6),
        ([], 2),
    ],
)
def test_resize(state, N):
    State = stateStr(state)
    nnz = State.nnz()
    State.resize(N)

    if N < len(state):
        # check it has the correct size
        assert State.size() == N
        # and number of non-zero entries
        assert State.nnz() == 0
        # and represents the original state
        assert State.state() == [0 for i in range(N)]

    else:
        # check it has the correct size
        assert State.size() == N
        # and number of non-zero entries
        assert State.nnz() == nnz
        # and represents the original state
        assert State.state() == state + [0 for i in range(N - len(state))]


@pytest.mark.parametrize(
    "state,coeff",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0),
        ([0, 0, 0, 0, 0], 2.0+1.0j),
        ([], 5.0),
    ],
)
def test_rmul(state, coeff):
    State = stateStr(state)
    State2 = coeff*State

    # check that the result of the multiplication is correct
    assert isSepState(State2)
    print(type(State2.state()))
    print(type(State))
    assert State2.state() == State
    assert State2.size() == State.size()
    assert State2.nnz() == State.nnz()

    assert State2.coeff == coeff


@pytest.mark.parametrize(
    "state,coeff",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0),
        ([0, 0, 0, 0, 0], 2.0+1.0j),
        ([], 5.0),
    ],
)
def test_mul(state, coeff):
    State = stateStr(state)
    State2 = State*coeff

    # check that the result of the multiplication is correct
    assert isSepState(State2)
    assert State2.state() == State
    assert State2.size() == State.size()
    assert State2.nnz() == State.nnz()

    assert State2.coeff == coeff


@pytest.mark.parametrize(
    "state,coeff",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], 1.0),
        ([0, 0, 0, 0, 0], 2.0+1.0j),
        ([], 5.0),
    ],
)
def test_div(state, coeff):
    State = stateStr(state)
    State2 = State/coeff

    # check that the result of the multiplication is correct
    assert isSepState(State2)
    assert State2.state() == State
    assert State2.size() == State.size()
    assert State2.nnz() == State.nnz()

    assert State2.coeff == 1.0/coeff


@pytest.mark.parametrize(
    "state1,state2",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], [1, 0, 2, 2, 3, 4, 3, 2]),
        ([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]),
        ([], []),
    ],
)
def test_add(state1, state2):
    State = stateStr(state1)
    State2 = stateStr(state2)

    State3 = State+State2

    # check that the result of the multiplication is correct
    assert isKet(State3)

@pytest.mark.parametrize(
    "state1,state2",
    [
        ([1, 0, 2, 0, 3, 4, 0, 2], [1, 0, 2, 2, 3, 4, 3, 2]),
        ([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]),
        ([], []),
    ],
)
def test_sub(state1, state2):
    State = stateStr(state1)
    State2 = stateStr(state2)

    State3 = State-State2

    # check that the result of the multiplication is correct
    assert isKet(State3)
