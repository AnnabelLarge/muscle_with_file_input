"""Test 2 of the plan: MUSCLE's own posteriors, dumped and read back.

``-postout D`` then ``-postdir D -postmap D/pair_map.tsv -postfamily F``
must give the same .afa as the built-in run, byte for byte, for every
option combination below. The dump itself must not depend on the
thread count, and reading it back with -postout again must reproduce it.
"""

import itertools
import os

import numpy as np
import pytest

from conftest import muscle, read_bytes, write_fasta

FAMILIES = ['n2', 'n3', 'n5dup', 'n12']
COMBOS = list(itertools.product((0, 2), (0, 100), (None, 'abc'), (1, 8)))


def opts(consiters, refineiters, perm, threads):
    out = ['-consiters', consiters, '-refineiters', refineiters,
           '-threads', threads]
    if perm is not None:
        out += ['-perm', perm]
    return out


def post_in(d, family):
    return ['-postdir', d, '-postmap', os.path.join(d, 'pair_map.tsv'),
            '-postfamily', family]


def dir_bytes(d):
    """``{relative path: bytes}`` of every file under ``d``."""
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, d)] = read_bytes(p)
    return out


@pytest.fixture(scope='session')
def dumps(muscle_bin, fasta_dir, tmp_path_factory):
    """``{family: dump dir}``, from a 1-thread -postout run."""
    root = tmp_path_factory.mktemp('dumps')
    out = {}
    for fam in FAMILIES:
        d = root / fam
        muscle('-align', fasta_dir / f'{fam}.fa', '-postout', d,
               '-postfamily', fam, '-output', root / f'{fam}.afa',
               '-threads', 1)
        out[fam] = d
    return out


@pytest.mark.parametrize('family', FAMILIES)
def test_dump_layout(dumps, fasta_dir, families, family):
    d = dumps[family]
    assert read_bytes(d / 'seqs.fasta') == read_bytes(fasta_dir /
                                                      f'{family}.fa')
    lines = (d / 'pair_map.tsv').read_text().splitlines()
    assert lines[0] == 'family\trow_label\tcol_label\tLX\tLY\tfile\ttranspose'
    names = [nm for nm, _ in families[family]]
    seqs = dict(families[family])
    # MUSCLE dereplicates: only pairs of first copies, i < j, in order
    reps, seen = [], set()
    for nm in names:
        if seqs[nm] not in seen:
            seen.add(seqs[nm])
            reps.append(nm)
    want = [(a, b) for k, a in enumerate(reps) for b in reps[k + 1:]]
    rows = [ln.split('\t') for ln in lines[1:]]
    assert [(r[1], r[2]) for r in rows] == want
    for fam, row, col, lx, ly, file, tr in rows:
        i, j = names.index(row), names.index(col)
        assert (fam, file, tr) == (family, f'pairs/{i}_{j}.npy', '0')
        P = np.load(d / file)
        assert P.dtype == np.float32 and P.shape == (len(seqs[row]),
                                                     len(seqs[col]))
        assert P.shape == (int(lx), int(ly))
        assert np.all(np.isfinite(P)) and P.min() >= 0
        assert not np.any((P > 0) & (P < 0.01))   # CalcPostFlat's threshold
    assert sorted(os.listdir(d / 'pairs')) == sorted(
        os.path.basename(r[5]) for r in rows)


@pytest.mark.parametrize('family', FAMILIES)
def test_dump_does_not_depend_on_threads(dumps, fasta_dir, tmp_path, family):
    d = tmp_path / 'd8'
    muscle('-align', fasta_dir / f'{family}.fa', '-postout', d,
           '-postfamily', family, '-output', tmp_path / 'x.afa',
           '-threads', 8)
    assert dir_bytes(d) == dir_bytes(dumps[family])


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('combo', COMBOS,
                         ids=['c%s-r%s-p%s-t%s' % c for c in COMBOS])
def test_round_trip_identical_msa(dumps, fasta_dir, tmp_path, family, combo):
    fa = fasta_dir / f'{family}.fa'
    a, b = tmp_path / 'builtin.afa', tmp_path / 'postin.afa'
    muscle('-align', fa, '-output', a, *opts(*combo))
    p = muscle('-align', fa, '-output', b, *post_in(dumps[family], family),
               *opts(*combo))
    assert 'Posteriors: from' in p.stderr
    assert read_bytes(a) == read_bytes(b)


@pytest.mark.parametrize('family', FAMILIES)
def test_reading_then_dumping_reproduces_the_dump(dumps, fasta_dir, tmp_path,
                                                  family):
    """-postdir plus -postout: what MUSCLE read is what it wrote."""
    d = tmp_path / 'again'
    muscle('-align', fasta_dir / f'{family}.fa', '-output', tmp_path / 'x.afa',
           *post_in(dumps[family], family), '-postout', d,
           '-postfamily', family)
    assert dir_bytes(d) == dir_bytes(dumps[family])


def test_one_unique_sequence(muscle_bin, tmp_path):
    """All sequences identical: no pairs; the dump has an empty map and
    reading it back works."""
    fa = tmp_path / 'same.fa'
    write_fasta(fa, [('s0', 'ACDEFGHIKLMNPQRSTVWY'),
                     ('s1', 'ACDEFGHIKLMNPQRSTVWY')])
    d = tmp_path / 'd'
    muscle('-align', fa, '-postout', d, '-postfamily', 'same',
           '-output', tmp_path / 'a.afa')
    assert (d / 'pair_map.tsv').read_text().count('\n') == 1
    muscle('-align', fa, *post_in(d, 'same'), '-output', tmp_path / 'b.afa')
    assert read_bytes(tmp_path / 'a.afa') == read_bytes(tmp_path / 'b.afa')
