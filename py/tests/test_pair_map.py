"""Tests of the preprocessing for `muscle -postdir/-postmap` (no MUSCLE run).

The map written by ``write_pair_map`` and read back the way MUSCLE will
(``load_pairs_from_map``) must give, for every pair i < j, exactly the
matrix that ``prepare.load_family`` gives, transposed when its key is
(j, i).

- toy: a layout from the FSA workflow's ``tests/layout.py``, where two
  families share sequences, so a pair of one family uses the
  deduplicated posterior of the other, in both orientations;
- real: sabre-twi and prefab with the neuralTKF posteriors on /mnt/sdb
  (or ``$FSA_REAL_POSTERIORS``); skipped when not present.
"""

import os
import sys

import numpy as np
import pytest

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PY_DIR)

from muscle_posteriors import (FSA_SCORING, family_fasta_seqs,  # noqa: E402
                               load_pairs_from_map, read_pair_map,
                               write_family_fasta, write_pair_map)

sys.path.insert(0, os.path.join(FSA_SCORING, 'tests'))
from prepare import load_family, prepare_dataset  # noqa: E402
from run_fsa import residues_match  # noqa: E402

REAL_ROOT = (os.environ.get('FSA_REAL_POSTERIORS')
             or '/mnt/sdb/annabel/2d_posteriors_bali3_ox_prefab_sabre')


def expected_pairs(fam):
    """``load_family``'s matrices, keyed i < j, rows = sequence i."""
    out = {}
    for (i, j), P in fam['pair_posteriors'].items():
        out[(min(i, j), max(i, j))] = P if i < j else P.T
    return out


def check_family(prep, pair_map, family, tmp_path):
    fam = load_family(prep, family)
    got = load_pairs_from_map(pair_map, prep['cache_dir'], family,
                              fam['names'])
    want = expected_pairs(fam)
    assert sorted(got) == sorted(want)
    for k, P in want.items():
        assert got[k].dtype == np.float32
        assert np.array_equal(got[k], P), (family, k)
    # seqs.fasta: lengths match the map, and it passes the reference check
    seqs = dict(family_fasta_seqs(fam))
    assert list(seqs) == fam['names']
    for (row, col), (lx, ly, _, _) in pair_map[family].items():
        assert (len(seqs[row]), len(seqs[col])) == (lx, ly)
    assert set(''.join(seqs.values())) <= set('ACDEFGHIKLMNPQRSTVWYX')
    for nm in fam['names']:          # what check_msa accepts per residue
        raw = fam['raw_seqs'][nm].upper()
        assert len(seqs[nm]) == len(raw)
        assert all(residues_match(a, b) for a, b in zip(seqs[nm], raw))
    path = tmp_path / f'{family}.fasta'
    write_family_fasta(fam, str(path))
    lines = path.read_text().splitlines()
    assert lines[0::2] == [f'>{nm}' for nm in fam['names']]
    assert lines[1::2] == list(seqs.values())
    return fam


def map_stats(pair_map):
    """Number of lines, transposed lines, and files used by >1 family."""
    users = {}
    n = n_tr = 0
    for family, pairs in pair_map.items():
        for (lx, ly, file, tr) in pairs.values():
            n += 1
            n_tr += tr
            users.setdefault(file, set()).add(family)
    shared = sum(1 for fams in users.values() if len(fams) > 1)
    return n, n_tr, len(users), shared


# ---- toy layout ---------------------------------------------------------

def toy_families(rng):
    aa = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    s = [''.join(rng.choice(aa, size=int(rng.integers(20, 60))))
         for _ in range(5)]
    # BB2 reuses BB1's sequences under other names and in another order,
    # so its pairs map to BB1's deduplicated rows (some reversed); BB1 has
    # a lowercase sequence and B/Z, BB3 two identical sequences.
    return {
        'BB1': [('p', s[0]), ('q', s[1].lower()), ('r', s[2] + 'BZ')],
        'BB2': [('u', s[2] + 'BZ'), ('v', s[0]), ('w', s[3])],
        'BB3': [('x', s[4]), ('y', s[3]), ('z', s[4])],
    }


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    from conftest import toy_layout  # the FSA workflow's tests/conftest.py
    root = tmp_path_factory.mktemp('toy')
    rng = np.random.default_rng(3)
    paths = toy_layout(root, toy_families(rng), seed=1)
    prep = prepare_dataset('bali3', paths['posterior_root'],
                           str(root / 'work'),
                           data_dir=paths['data_dir'],
                           one_orientation_dir=paths['one_orientation_dir'],
                           references_dir=paths['references_dir'])
    path = str(root / 'pair_map.tsv')
    write_pair_map(prep, path)
    return prep, path


def test_toy_map_matches_load_family(toy, tmp_path):
    prep, path = toy
    pair_map = read_pair_map(path)
    assert sorted(pair_map) == ['BB1', 'BB2', 'BB3']
    for family in prep['families']:
        check_family(prep, pair_map, family, tmp_path)
    n, n_tr, n_files, shared = map_stats(pair_map)
    assert n == 9
    assert 0 < n_tr < n          # both orientations occur
    assert shared >= 1           # BB2 uses BB1's posteriors
    assert n_files < n           # and BB3's duplicate shares one file


def test_toy_map_is_deterministic(toy, tmp_path):
    prep, path = toy
    other = str(tmp_path / 'again.tsv')
    write_pair_map(prep, other, families=list(reversed(prep['families'])))
    assert open(path, 'rb').read() == open(other, 'rb').read()


def test_toy_transpose_flag_is_orientation(toy):
    """transpose is 1 exactly when the posterior's ancestor is col_label."""
    prep, path = toy
    pair_map = read_pair_map(path)
    for family in prep['families']:
        rec = prep['records'][family]
        for prov in rec['provenance']:
            a, d = prov['ancestor'], prov['descendant']
            if (a, d) in pair_map[family]:
                assert pair_map[family][(a, d)][3] == 0
            else:
                assert pair_map[family][(d, a)][3] == 1


# ---- real data ----------------------------------------------------------
# sabre-twi: 108 families, no shared posteriors. prefab: 1681 pairs, 18 of
# them use the posterior of another family's pair.

REAL = {'sabre-twi': ('neuralTKF_sabre', 'SORTED-SABRE-TWI-DEDUP', False),
        'prefab': ('neuralTKF_prefab4ref', 'SORTED-PREFAB4REF-DEDUP', True)}


@pytest.fixture(scope='module', params=sorted(REAL))
def real(request, tmp_path_factory):
    dataset = request.param
    sub, set_dir, shares = REAL[dataset]
    root = os.path.join(REAL_ROOT, sub)
    if not os.path.isdir(os.path.join(root, set_dir)):
        pytest.skip(f'no real posteriors under {root}')
    work = tmp_path_factory.mktemp(dataset)
    prep = prepare_dataset(dataset, root, str(work), n_workers=8)
    path = str(work / 'pair_map.tsv')
    write_pair_map(prep, path)
    return dataset, shares, prep, path


def shared_families(pair_map):
    """Families with a file that another family also uses."""
    users = {}
    for family, pairs in pair_map.items():
        for v in pairs.values():
            users.setdefault(v[2], set()).add(family)
    return sorted({f for fams in users.values() if len(fams) > 1
                   for f in fams})


def test_real_map_matches_load_family(real, tmp_path):
    dataset, shares, prep, path = real
    pair_map = read_pair_map(path)
    assert sorted(pair_map) == sorted(prep['families'])
    size = lambda f: len(prep['records'][f]['names'])
    check = sorted(prep['families'], key=size)[:5]
    sharing = shared_families(pair_map)
    assert bool(sharing) == shares
    check += sorted(sharing, key=size)[:3]          # small sharing families
    for family in check:
        check_family(prep, pair_map, family, tmp_path)
    n, n_tr, n_files, shared = map_stats(pair_map)
    print(f'\n{dataset}: {len(pair_map)} families, {n} map lines, '
          f'{n_tr} transposed, {n_files} distinct files, {shared} files '
          f'used by more than one family; checked {check}')
    assert 0 < n_tr < n


def test_real_every_file_is_strict_npy(real):
    """Every file the map names passes the checks MUSCLE's reader makes."""
    from muscle_posteriors import read_npy_strict
    dataset, shares, prep, path = real
    pair_map = read_pair_map(path)
    files = {v[2] for pairs in pair_map.values() for v in pairs.values()}
    for file in sorted(files):
        read_npy_strict(os.path.join(prep['cache_dir'], file))
