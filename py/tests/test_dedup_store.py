"""Test 3 of the plan: a shared dedup folder + map, through the Python API.

Three families share sequences (taken from the generated n12 family):

- fA: d0..d5;
- fB: d4, a new sequence, d1, a copy of d4, d2. Its pairs (d4, d1) and
  (d4, d2) have the other orientation than in fA; (d1, d2) the same;
- fC: d1, d2, d5, all in fA's order.

From MUSCLE's own dumps, a dedup folder is built in which every pair of
sequences is stored once (shared within and across families), every
other new matrix stored transposed, with one map for all families. Then
``write_family_fasta`` + ``run_muscle`` + ``read_msa``.

MUSCLE's pair HMM is not exactly symmetric (the posterior of (b, a) is
not bit for bit the transpose of (a, b)), so a family can be compared
with its built-in run only where no pair it needs is taken from the other
orientation:

- fA and fC: byte-identical to the built-in run;
- every family: byte-identical to a run on a plain per-family folder
  holding the matrices that the map implies (no sharing, no transpose).
"""

import os

import numpy as np
import pytest

from conftest import muscle, read_bytes, write_fasta, mutate
from muscle_posteriors import (load_pairs_from_map, read_afa, read_msa,
                               read_pair_map, run_muscle, write_family_fasta)
from vendored.scoring import encode_seq

HEADER = 'family\trow_label\tcol_label\tLX\tLY\tfile\ttranspose'


def make_fams(families):
    d = dict(families['n12'])
    rng = np.random.default_rng(7)
    seqs = {
        'fA': [(f'd{k}', d[f'd{k}']) for k in range(6)],
        'fB': [('x0', d['d4']), ('x1', mutate(rng, d['d3'])),
               ('x2', d['d1']), ('x3', d['d4']), ('x4', d['d2'])],
        'fC': [('y0', d['d1']), ('y1', d['d2']), ('y2', d['d5'])],
    }
    fams = {}
    for f, recs in seqs.items():
        names = [nm for nm, _ in recs]
        fams[f] = {'names': names,
                   'int_seqs': {nm: encode_seq(s) for nm, s in recs},
                   'raw_seqs': dict(recs)}
    return fams


def read_dump(d):
    out = {}
    for line in (d / 'pair_map.tsv').read_text().splitlines()[1:]:
        _, row, col, _, _, file, _ = line.split('\t')
        out[(row, col)] = np.load(d / file)
    return out


def build_store(fams, dumps, post_dir):
    """One file per pair of sequences; every other new file transposed.
    Returns the map lines (all n(n-1)/2 pairs of each family)."""
    os.makedirs(post_dir)
    stored = {}      # (seq of the file's rows, seq of its cols) -> file
    lines, k = [], 0
    for f in sorted(fams):
        names, seqs = fams[f]['names'], fams[f]['raw_seqs']
        rep = {}
        for nm in names:
            rep[nm] = next(m for m in names if seqs[m] == seqs[nm])
        for i, row in enumerate(names):
            for col in names[i + 1:]:
                key = (seqs[row], seqs[col])
                if key in stored:
                    file, tr = stored[key], 0
                elif key[::-1] in stored:
                    file, tr = stored[key[::-1]], 1
                else:
                    ri, rj = rep[row], rep[col]
                    if ri == rj:          # identical sequences: never read
                        P = np.zeros((len(seqs[row]),) * 2, np.float32)
                    elif names.index(ri) < names.index(rj):
                        P = dumps[f][(ri, rj)]
                    else:
                        P = dumps[f][(rj, ri)].T
                    k += 1
                    file = f'm{k}.npy'
                    if k % 2:
                        np.save(post_dir / file, np.ascontiguousarray(P.T))
                        stored[key[::-1]], tr = file, 1
                    else:
                        np.save(post_dir / file, np.ascontiguousarray(P))
                        stored[key], tr = file, 0
                lines.append((f, row, col, len(seqs[row]), len(seqs[col]),
                              file, tr))
    return lines


def write_map(path, lines):
    with open(path, 'w') as fh:
        fh.write(HEADER + '\n')
        for ln in lines:
            fh.write('\t'.join(map(str, ln)) + '\n')


@pytest.fixture(scope='module')
def setup(muscle_bin, families, tmp_path_factory):
    root = tmp_path_factory.mktemp('store')
    fams = make_fams(families)
    dumps, builtin = {}, {}
    for f, fam in fams.items():
        fa = root / f'{f}.fasta'
        write_family_fasta(fam, str(fa))
        muscle('-align', fa, '-postout', root / f'dump_{f}', '-postfamily', f,
               '-input_order', '-output', root / f'{f}.builtin.afa')
        dumps[f] = read_dump(root / f'dump_{f}')
        builtin[f] = root / f'{f}.builtin.afa'
    lines = build_store(fams, dumps, root / 'store')
    write_map(root / 'pair_map.tsv', lines)
    return root, fams, dumps, builtin, lines


def test_store_shares_and_transposes(setup):
    root, fams, dumps, builtin, lines = setup
    files = {}
    for f, row, col, lx, ly, file, tr in lines:
        files.setdefault(file, set()).add(f)
    assert len(lines) == 15 + 10 + 3
    assert len(files) < len(lines)
    assert any(len(v) > 1 for v in files.values())       # across families
    assert {tr for *_, tr in lines} == {0, 1}


def test_map_gives_the_dumps_where_orientation_is_kept(setup):
    root, fams, dumps, builtin, lines = setup
    pm = read_pair_map(str(root / 'pair_map.tsv'))
    for f in ('fA', 'fC'):
        got = load_pairs_from_map(pm, str(root / 'store'), f,
                                  fams[f]['names'])
        names = fams[f]['names']
        for (row, col), P in dumps[f].items():
            k = (names.index(row), names.index(col))
            assert np.array_equal(got[k], P), (f, row, col)


@pytest.mark.parametrize('f', ['fA', 'fB', 'fC'])
def test_msa_through_the_store(setup, muscle_bin, tmp_path, f):
    root, fams, dumps, builtin, lines = setup
    fam = fams[f]
    fa = tmp_path / 'seqs.fasta'
    write_family_fasta(fam, str(fa))
    out = tmp_path / 'store.afa'
    err = run_muscle(fa, out, muscle_bin, root / 'store',
                     root / 'pair_map.tsv', f)
    assert f'family {f}' in err
    msa = read_msa(str(out), fam)
    assert list(msa) == fam['names']        # read_msa reorders

    # reference: a plain folder with the matrices the map implies
    pm = read_pair_map(str(root / 'pair_map.tsv'))
    mats = load_pairs_from_map(pm, str(root / 'store'), f, fam['names'])
    plain = tmp_path / 'plain'
    os.makedirs(plain)
    plain_lines = []
    for (i, j), P in sorted(mats.items()):
        np.save(plain / f'{i}_{j}.npy', P)
        row, col = fam['names'][i], fam['names'][j]
        plain_lines.append((f, row, col, P.shape[0], P.shape[1],
                            f'{i}_{j}.npy', 0))
    write_map(plain / 'pair_map.tsv', plain_lines)
    out2 = tmp_path / 'plain.afa'
    run_muscle(fa, out2, muscle_bin, plain, plain / 'pair_map.tsv', f)
    assert read_bytes(out) == read_bytes(out2)

    if f in ('fA', 'fC'):
        assert read_bytes(out) == read_bytes(builtin[f])
