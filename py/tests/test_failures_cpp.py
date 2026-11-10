"""Test 5 of the plan: bad -postdir / -postmap inputs stop MUSCLE.

Each case edits a copy of MUSCLE's own dump of n5dup (pairs of c0, c1,
c2, c4; c3 is a copy of c1) and must make MUSCLE exit non-zero with a
message that names the problem. Two cases check what is accepted: lines
MUSCLE does not need are ignored and counted, and values below 0.01 are
set to 0 on reading.
"""

import os
import shutil

import numpy as np
import pytest

from conftest import muscle, read_bytes

FAMILY = 'n5dup'
HEADER = 'family\trow_label\tcol_label\tLX\tLY\tfile\ttranspose'


@pytest.fixture(scope='module')
def dump(muscle_bin, fasta_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp('dump')
    d = root / 'd'
    muscle('-align', fasta_dir / f'{FAMILY}.fa', '-postout', d,
           '-postfamily', FAMILY, '-output', root / 'builtin.afa')
    return d, root / 'builtin.afa'


@pytest.fixture
def case(dump, tmp_path):
    """A fresh copy of the dump: (dir, map lines as field lists)."""
    d = tmp_path / 'd'
    shutil.copytree(dump[0], d)
    lines = (d / 'pair_map.tsv').read_text().splitlines()
    assert lines[0] == HEADER
    return d, [ln.split('\t') for ln in lines[1:]]


def write_map(d, rows, header=HEADER):
    with open(d / 'pair_map.tsv', 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')


def run(fasta_dir, tmp_path, d, *extra, family=FAMILY, postin=True):
    args = ['-align', fasta_dir / f'{FAMILY}.fa',
            '-output', tmp_path / 'out.afa']
    if postin:
        args += ['-postdir', d, '-postmap', d / 'pair_map.tsv',
                 '-postfamily', family]
    return muscle(*args, *extra, check=False)


def assert_dies(p, message):
    assert p.returncode != 0
    assert '---Fatal error---' in p.stderr
    assert message in p.stderr, p.stderr[-1000:]


def edit_npy(d, row, fn):
    path = d / row[5]
    np.save(path, fn(np.load(path)), allow_pickle=False)


# ---- the map -------------------------------------------------------------

def test_missing_pair(case, fasta_dir, tmp_path):
    d, rows = case
    write_map(d, rows[1:])
    assert_dies(run(fasta_dir, tmp_path, d), 'no line for pair c0, c1')


def test_family_not_in_map(case, fasta_dir, tmp_path):
    d, rows = case
    assert_dies(run(fasta_dir, tmp_path, d, family='n5'),
                'has no lines for family n5')


def test_swapped_orientation(case, fasta_dir, tmp_path):
    """The same matrix, described as (c1, c0) with transpose 1: the
    orientation must be chosen in Python, so MUSCLE refuses it."""
    d, rows = case
    fam, row, col, lx, ly, file, _ = rows[0]
    rows[0] = [fam, col, row, ly, lx, file, 1]
    write_map(d, rows)
    assert_dies(run(fasta_dir, tmp_path, d),
                'row_label must be the sequence that comes first')


def test_wrong_label(case, fasta_dir, tmp_path):
    d, rows = case
    rows[2][2] = rows[2][2] + 'x'
    write_map(d, rows)
    assert_dies(run(fasta_dir, tmp_path, d), 'no line for pair')


def test_wrong_lengths_in_map(case, fasta_dir, tmp_path):
    d, rows = case
    rows[0][4] = int(rows[0][4]) + 1
    write_map(d, rows)
    assert_dies(run(fasta_dir, tmp_path, d), 'in the map')


def test_wrong_transpose_flag(case, fasta_dir, tmp_path):
    d, rows = case
    assert rows[0][3] != rows[0][4]          # not square
    rows[0][6] = 1
    write_map(d, rows)
    assert_dies(run(fasta_dir, tmp_path, d), 'expected (')


@pytest.mark.parametrize('bad,message', [
    ('2', 'transpose must be 0 or 1'),
    ('', 'transpose must be 0 or 1'),
])
def test_bad_transpose_value(case, fasta_dir, tmp_path, bad, message):
    d, rows = case
    rows[1][6] = bad
    write_map(d, rows)
    assert_dies(run(fasta_dir, tmp_path, d), message)


def test_repeated_pair(case, fasta_dir, tmp_path):
    d, rows = case
    write_map(d, rows + [rows[0]])
    assert_dies(run(fasta_dir, tmp_path, d), 'twice')


def test_bad_header(case, fasta_dir, tmp_path):
    d, rows = case
    write_map(d, rows, header=HEADER.replace('transpose', 'flip'))
    assert_dies(run(fasta_dir, tmp_path, d), 'header must be')


def test_wrong_field_count(case, fasta_dir, tmp_path):
    d, rows = case
    rows[3] = rows[3][:6]
    write_map(d, rows)
    assert_dies(run(fasta_dir, tmp_path, d), '6 fields, expected 7')


def test_missing_map(case, fasta_dir, tmp_path):
    d, rows = case
    os.remove(d / 'pair_map.tsv')
    assert_dies(run(fasta_dir, tmp_path, d), '-postmap: cannot open')


# ---- the matrices --------------------------------------------------------

def test_missing_file(case, fasta_dir, tmp_path):
    d, rows = case
    os.remove(d / rows[4][5])
    assert_dies(run(fasta_dir, tmp_path, d), 'Cannot open posterior file')


def test_wrong_shape(case, fasta_dir, tmp_path):
    d, rows = case
    edit_npy(d, rows[0], lambda P: P[:, :-1].copy())
    assert_dies(run(fasta_dir, tmp_path, d), 'expected (')


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_non_finite(case, fasta_dir, tmp_path, value):
    d, rows = case

    def f(P):
        P[3, 4] = value
        return P
    edit_npy(d, rows[1], f)
    assert_dies(run(fasta_dir, tmp_path, d), 'non-finite value at [3, 4]')


def test_negative(case, fasta_dir, tmp_path):
    d, rows = case

    def f(P):
        P[0, 2] = -0.5
        return P
    edit_npy(d, rows[1], f)
    assert_dies(run(fasta_dir, tmp_path, d), 'negative value -0.5 at [0, 2]')


@pytest.mark.parametrize('fn,message', [
    (lambda P: P.astype(np.float64), "dtype must be little-endian float32"),
    (lambda P: P.astype('>f4'), "dtype must be little-endian float32"),
    (lambda P: np.asfortranarray(P), 'must be C order'),
    (lambda P: P[None], 'shape must have 2 axes'),
])
def test_bad_npy_format(case, fasta_dir, tmp_path, fn, message):
    d, rows = case
    edit_npy(d, rows[2], fn)
    assert_dies(run(fasta_dir, tmp_path, d), message)


def test_truncated_file(case, fasta_dir, tmp_path):
    d, rows = case
    path = d / rows[2][5]
    data = read_bytes(path)
    with open(path, 'wb') as f:
        f.write(data[:-4])
    assert_dies(run(fasta_dir, tmp_path, d), 'file size')


def test_not_npy(case, fasta_dir, tmp_path):
    d, rows = case
    (d / rows[2][5]).write_text('hello')
    assert_dies(run(fasta_dir, tmp_path, d), 'not a .npy file')


# ---- options -------------------------------------------------------------

@pytest.mark.parametrize('extra,message', [
    (['-perturb', 1], '-perturb has no effect with -postdir'),
    (['-stratified'], 'needs a single run'),
    (['-diversified'], 'needs a single run'),
    (['-replicates', 2], 'needs a single run'),
    (['-hmmin', 'none.hmm'], 'have no effect with -postdir'),
    (['-m_is', '0.1'], 'have no effect with -postdir'),
    (['-minsuper', 2], 'not supported with Super5'),
])
def test_rejected_options(case, fasta_dir, tmp_path, extra, message):
    d, rows = case
    assert_dies(run(fasta_dir, tmp_path, d, *extra), message)


def test_perturb_zero_is_allowed(case, dump, fasta_dir, tmp_path):
    d, rows = case
    p = run(fasta_dir, tmp_path, d, '-perturb', 0)
    assert p.returncode == 0, p.stderr[-1000:]
    assert read_bytes(tmp_path / 'out.afa') == read_bytes(dump[1])


@pytest.mark.parametrize('extra,message', [
    (['-postdir', 'x'], 'must be set together'),
    (['-postdir', 'x', '-postmap', 'x/pair_map.tsv'], 'must be set together'),
    (['-postmap', 'x/pair_map.tsv', '-postfamily', 'f'],
     'must be set together'),
    (['-postfamily', 'f'], '-postfamily needs -postdir or -postout'),
])
def test_incomplete_options(case, fasta_dir, tmp_path, extra, message):
    d, rows = case
    assert_dies(run(fasta_dir, tmp_path, d, *extra, postin=False), message)


def test_postout_same_as_postdir(case, fasta_dir, tmp_path):
    d, rows = case
    assert_dies(run(fasta_dir, tmp_path, d, '-postout', d), 'must differ')


# ---- accepted ------------------------------------------------------------

def test_unused_lines_are_ignored_and_counted(case, dump, fasta_dir,
                                              tmp_path):
    """Lines for the duplicate c3 and for another family are not needed."""
    d, rows = case
    extra = [[FAMILY, 'c1', 'c3', rows[0][4], rows[0][4], rows[0][5], 0],
             [FAMILY, 'c3', 'c4', 1, 1, 'nowhere.npy', 0],
             ['other', 'c0', 'c1', 1, 1, 'nowhere.npy', 1]]
    write_map(d, extra[2:] + rows + extra[:2])
    p = run(fasta_dir, tmp_path, d)
    assert p.returncode == 0, p.stderr[-1000:]
    assert '6 pairs from' in p.stderr
    assert '2 unused lines of family n5dup' in p.stderr
    assert read_bytes(tmp_path / 'out.afa') == read_bytes(dump[1])


def test_values_below_threshold_are_zeroed(case, dump, fasta_dir, tmp_path):
    """Adding values < 0.01 where the dump has zeros changes nothing."""
    d, rows = case
    for r in rows:
        edit_npy(d, r, lambda P: np.where(P == 0, np.float32(0.0099),
                                          P).astype(np.float32))
    p = run(fasta_dir, tmp_path, d)
    assert p.returncode == 0, p.stderr[-1000:]
    assert read_bytes(tmp_path / 'out.afa') == read_bytes(dump[1])
