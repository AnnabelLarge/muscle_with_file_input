"""Test 4 of the plan: the real stage 0-1 pipeline, then MUSCLE.

A toy bali3 dataset is written with the FSA workflow's
``tests/layout.py`` (``write_layout``, unchanged). Its posterior samples
are MUSCLE's own dumps, stored in the orientation of each kept row
(FW: rows = the pair's first sequence, RV: the transpose). Then
``run_muscle.run_dataset``: prepare_dataset (stages 0-1, cache = the
dedup folder) -> write_pair_map -> load_family -> write_family_fasta ->
MUSCLE -> read_msa -> save_msa -> check_all_saved. Each family's .afa
must equal the built-in run on the same (converted) FASTA, byte for byte.

The raw input has lowercase letters and B/Z (converted to upper case and
X, as encode_seq does); BBn5dup has a duplicate sequence; BBshare uses
two of BBn12's sequences in the same order, so one of its pairs uses a
posterior of BBn12's (or the other way round).
"""

import itertools
import json
import os
import re
import sys

import numpy as np
import pytest

from conftest import MUSCLE_BIN, muscle, read_bytes, write_fasta
from muscle_posteriors import FSA_SCORING, read_pair_map

sys.path.append(os.path.join(FSA_SCORING, 'tests'))
from layout import random_posterior, write_layout  # noqa: E402

import run_muscle  # noqa: E402

SETS = ('BALI3-SHORT', 'BALI3-MEDIUM', 'BALI3-LONG')
PAIR_ID = re.compile(r'^(FW|RV)_(.+)_pair_(\d+)$')


def toy_families(families):
    """``(raw, converted)``: ``{family: [(name, seq)]}`` each."""
    d12 = dict(families['n12'])
    conv = {
        'BBn2': families['n2'],
        'BBn3': families['n3'],
        'BBn5dup': families['n5dup'],
        'BBn12': families['n12'],
        'BBshare': [('e0', d12['d0']), ('e1', families['n3'][0][1]),
                    ('e2', d12['d3'])],
    }
    raw = {f: list(v) for f, v in conv.items()}
    # lowercase, and B / Z where the converted sequence has X
    raw['BBn2'][1] = ('a1', raw['BBn2'][1][1].lower())
    raw['BBshare'][0] = ('e0', d12['d0'].lower())
    nm, s = raw['BBn12'][2]
    xs = [i for i, c in enumerate(s) if c == 'X']
    assert len(xs) == 3
    s = list(s)
    s[xs[0]], s[xs[1]] = 'B', 'z'
    raw['BBn12'][2] = (nm, ''.join(s))
    return raw, conv


def read_dump(d):
    out = {}
    for line in open(os.path.join(d, 'pair_map.tsv')).read().splitlines()[1:]:
        _, row, col, _, _, file, _ = line.split('\t')
        out[(row, col)] = np.load(os.path.join(d, file))
    return out


@pytest.fixture(scope='module')
def run(muscle_bin, families, tmp_path_factory):
    root = tmp_path_factory.mktemp('pipeline')
    raw, conv = toy_families(families)

    builtin, dumps = {}, {}
    for f, seqs in conv.items():
        fa = root / f'{f}.conv.fasta'
        write_fasta(fa, seqs)
        muscle('-align', fa, '-postout', root / f'dump_{f}', '-postfamily', f,
               '-input_order', '-output', root / f'{f}.builtin.afa')
        builtin[f] = root / f'{f}.builtin.afa'
        dumps[f] = read_dump(root / f'dump_{f}')

    def matrix(f, a, b):
        """Posterior with rows = sequence a of family f (any order)."""
        names = [nm for nm, _ in conv[f]]
        seqs = dict(conv[f])
        rep = {nm: next(m for m in names if seqs[m] == seqs[nm])
               for nm in names}
        ra, rb = rep[names[a]], rep[names[b]]
        if ra == rb:                      # identical sequences: never read
            return random_posterior(np.random.default_rng(0),
                                    (len(seqs[ra]),) * 2)
        if names.index(ra) < names.index(rb):
            return dumps[f][(ra, rb)]
        return dumps[f][(rb, ra)].T

    def posterior(pid, anc_len, desc_len, rng):
        fw, f, n = PAIR_ID.match(pid).groups()
        a, b = list(itertools.combinations(range(len(conv[f])), 2))[int(n)]
        M = matrix(f, a, b) if fw == 'FW' else matrix(f, b, a)
        assert M.shape == (anc_len, desc_len), (pid, M.shape)
        return M

    paths = write_layout(str(root / 'layout'), 'bali3', SETS, 'bali3', raw,
                         posterior=posterior, seed=2)
    out = root / 'out'
    summary = run_muscle.run_dataset(
        'bali3', paths['posterior_root'], str(out), muscle_bin=MUSCLE_BIN,
        data_dir=paths['data_dir'],
        one_orientation_dir=paths['one_orientation_dir'],
        references_dir=paths['references_dir'], jobs=2)
    return out, summary, builtin, conv, raw


FAMS = ['BBn12', 'BBn2', 'BBn3', 'BBn5dup', 'BBshare']


@pytest.mark.parametrize('family', FAMS)
def test_same_msa_as_builtin(run, family):
    out, summary, builtin, conv, raw = run
    afa = out / 'muscle' / family / 'muscle.afa'
    assert read_bytes(afa) == read_bytes(builtin[family])
    # MUSCLE read the converted sequences
    fasta = (out / 'muscle' / family / 'seqs.fasta').read_text()
    assert fasta == ''.join(f'>{nm}\n{s}\n' for nm, s in conv[family])


@pytest.mark.parametrize('family', FAMS)
def test_saved_msa(run, family):
    """msas/muscle/<family>.fasta: input order, rows = raw input, gaps."""
    out, summary, builtin, conv, raw = run
    text = (out / 'msas' / 'muscle' / f'{family}.fasta').read_text()
    lines = text.splitlines()
    assert lines[0::2] == [f'>{nm}' for nm, _ in raw[family]]
    rows = lines[1::2]
    assert len({len(r) for r in rows}) == 1
    assert [r.replace('-', '') for r in rows] == [s for _, s in conv[family]]


def test_map_shares_and_transposes(run):
    out, summary, builtin, conv, raw = run
    pm = read_pair_map(str(out / 'pair_map.tsv'))
    assert sorted(pm) == FAMS
    users = {}
    for f, pairs in pm.items():
        for v in pairs.values():
            users.setdefault(v[2], set()).add(f)
    assert {'BBn12', 'BBshare'} in [u for u in users.values() if len(u) > 1]
    assert {v[3] for pairs in pm.values() for v in pairs.values()} == {0, 1}
    log = (out / 'muscle' / 'BBn5dup' / 'muscle.log').read_text()
    assert '6 pairs from' in log and '4 unused lines of family BBn5dup' in log


def test_summary(run):
    out, summary, builtin, conv, raw = run
    on_disk = json.load(open(out / 'muscle_run.json'))
    assert on_disk['n_families'] == 5 == summary['n_families']
    assert sorted(on_disk['per_family']) == FAMS
    assert on_disk['muscle_version'].startswith('muscle ')
    assert on_disk['pair_map_lines'] == 1 + 3 + 10 + 66 + 3
