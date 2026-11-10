"""Shared pieces of the tests of `muscle -postout` / `-postdir`.

The MUSCLE binary is ``$MUSCLE_BIN``, or ``bin/muscle`` of this repo if
that is not set. Tests that need it skip when it does not exist.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PY_DIR)
sys.path.insert(0, PY_DIR)

MUSCLE_BIN = os.environ.get('MUSCLE_BIN') or os.path.join(REPO, 'bin',
                                                           'muscle')

AA = 'ACDEFGHIKLMNPQRSTVWY'
MIN_LEN, MAX_LEN = 20, 300


def random_seq(rng, length):
    return ''.join(rng.choice(list(AA), size=length))


def mutate(rng, seq, sub=0.2, indel=0.03):
    """Point mutations and short indels of ``seq``, length kept in range."""
    out = []
    for c in seq:
        r = rng.random()
        if r < indel / 2:
            continue                                   # deletion
        out.append(random_seq(rng, 1) if rng.random() < sub else c)
        if r > 1 - indel / 2:                          # insertion
            out.append(random_seq(rng, int(rng.integers(1, 4))))
    s = ''.join(out)
    if len(s) < MIN_LEN:
        s += random_seq(rng, MIN_LEN - len(s))
    return s[:MAX_LEN]


def related(rng, n, prefix, root_len):
    root = random_seq(rng, root_len)
    return [(f'{prefix}{k}', mutate(rng, root)) for k in range(n)]


def make_families(seed=0):
    """``{family: [(name, seq), ...]}``, FASTA order.

    - n2: two related sequences;
    - n3: three unrelated random sequences (flat posteriors);
    - n5dup: five related sequences, sequence 3 a copy of sequence 1
      (MUSCLE dereplicates it);
    - n12: twelve related sequences, lengths spread over 20-300, with a
      few X (wildcards) and one fragment of 20-29 residues.
    """
    rng = np.random.default_rng(seed)
    fams = {}
    fams['n2'] = related(rng, 2, 'a', 120)
    fams['n3'] = [(f'b{k}', random_seq(rng, int(rng.integers(MIN_LEN,
                                                             MAX_LEN + 1))))
                  for k in range(3)]
    n5 = related(rng, 5, 'c', 200)
    n5[3] = ('c3', n5[1][1])
    fams['n5dup'] = n5
    n12 = []
    root = random_seq(rng, 250)
    for k in range(12):
        s = mutate(rng, root, sub=0.3)
        # trim some to spread the lengths
        start = int(rng.integers(0, len(s) // 3)) if k % 3 == 1 else 0
        stop = len(s) - (int(rng.integers(0, len(s) // 2))
                         if k % 4 == 2 else 0)
        s = s[start:stop]
        if k == 10:                       # one short fragment
            s = s[40:40 + int(rng.integers(MIN_LEN, 30))]
        if len(s) < MIN_LEN:
            s = s + random_seq(rng, MIN_LEN - len(s))
        if k in (2, 7):
            pos = rng.choice(len(s), size=3, replace=False)
            s = ''.join('X' if i in set(pos) else c for i, c in enumerate(s))
        n12.append((f'd{k}', s))
    fams['n12'] = n12
    for fam, seqs in fams.items():
        for nm, s in seqs:
            assert MIN_LEN <= len(s) <= MAX_LEN, (fam, nm, len(s))
    return fams


def write_fasta(path, seqs):
    with open(path, 'w') as f:
        for nm, s in seqs:
            f.write(f'>{nm}\n{s}\n')


def muscle(*args, check=True, timeout=600):
    """Run MUSCLE; return the CompletedProcess (stdout and stderr as text)."""
    cmd = [MUSCLE_BIN, *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f'{" ".join(cmd)} exited {p.returncode}:\n'
                           f'{p.stderr[-3000:]}')
    return p


def read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


@pytest.fixture(scope='session')
def muscle_bin():
    if not os.path.exists(MUSCLE_BIN):
        pytest.skip(f'no MUSCLE binary at {MUSCLE_BIN} (set $MUSCLE_BIN)')
    return MUSCLE_BIN


@pytest.fixture(scope='session')
def families():
    return make_families()


@pytest.fixture(scope='session')
def fasta_dir(tmp_path_factory, families):
    """One FASTA per generated family: ``<dir>/<family>.fa``."""
    d = tmp_path_factory.mktemp('fasta')
    for fam, seqs in families.items():
        write_fasta(d / f'{fam}.fa', seqs)
    return d
