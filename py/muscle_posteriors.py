"""Inputs of `muscle -align -postdir DIR -postmap MAP -postfamily FAM`.

MUSCLE reads the posteriors from two inputs that cover a whole dataset:

1. ``DIR``: the deduplicated posteriors, one float32 ``.npy`` per kept
   pair (deduplicated across families and orientations). This is the
   stage-0 cache of the FSA workflow, ``<work_dir>/posterior_cache/``,
   with files ``SORTED-<S>-DEDUP/<kept_pairID>.npy``, used as it is.
2. ``MAP``: a plain TSV, one line per unordered pair of each family:

       family  row_label  col_label  LX  LY  file  transpose

   ``row_label`` is the name that comes first in ``fam['names']`` (and so
   in ``seqs.fasta``; MUSCLE's X), ``col_label`` the other one; ``LX``
   and ``LY`` their lengths. ``file`` is relative to ``DIR``. The stored
   matrix has shape ``(LX, LY)`` if ``transpose`` is 0, and ``(LY, LX)``
   if it is 1 (the posterior's ancestor is ``col_label``); MUSCLE then
   uses its transpose. Two lines can name the same file (pairs that share
   a deduplicated posterior, in one family or in several).

Per family MUSCLE also needs ``seqs.fasta`` (``write_family_fasta``).
``run_muscle`` runs it on one family and ``read_msa`` checks and reads
the result.

The map is built from the stage-1 records (``prepare.prepare_dataset``),
with no matrix read. ``load_pairs_from_map`` reads it back the way MUSCLE
will, for the tests.
"""

import os
import subprocess
import sys

import numpy as np

FSA_SCORING = os.environ.get('FSA_SCORING_DIR') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__)))), 'msa_with_new_decoder', 'fsa_scoring_on_ada_new_decoder')
if FSA_SCORING not in sys.path:
    sys.path.insert(0, FSA_SCORING)

from vendored.scoring import AA_CHARS  # noqa: E402

MAP_COLUMNS = ('family', 'row_label', 'col_label', 'LX', 'LY', 'file',
               'transpose')


class PairMapError(Exception):
    """The map cannot be written or does not match its inputs."""


def _check_name(family, name):
    if not name or any(c.isspace() for c in name):
        raise PairMapError(f'{family}: name {name!r} is empty or contains '
                           f'whitespace (the map is tab-separated and '
                           f'MUSCLE keeps the whole FASTA header line)')


def family_map_rows(family, record):
    """The map lines of one family, from its stage-1 record.

    Provenance keys are ``(idx[ancestor], idx[descendant])``; a key with
    i > j becomes ``transpose`` 1. Lines are in the record's pair order
    (sorted by the unordered pair).
    """
    names = record['names']
    _check_name(family, family)
    for nm in names:
        _check_name(family, nm)
    rows, seen = [], set()
    for prov in record['provenance']:
        i, j = prov['pair']
        if names[i] != prov['ancestor'] or names[j] != prov['descendant']:
            raise PairMapError(f'{family}: pair {(i, j)} does not match the '
                               f'names {prov["ancestor"]!r}, '
                               f'{prov["descendant"]!r}')
        if i == j or (min(i, j), max(i, j)) in seen:
            raise PairMapError(f'{family}: bad or repeated pair {(i, j)}')
        seen.add((min(i, j), max(i, j)))
        a, d = prov['anc_seq_len'], prov['desc_seq_len']
        if i < j:
            rows.append((family, names[i], names[j], a, d,
                         prov['cache_file'], 0))
        else:
            rows.append((family, names[j], names[i], d, a,
                         prov['cache_file'], 1))
    n = len(names)
    if len(seen) != n * (n - 1) // 2:
        raise PairMapError(f'{family}: {len(seen)} pairs for {n} sequences')
    return rows


def write_pair_map(prep, path, families=None):
    """Write the map of ``families`` (default all of ``prep``) to ``path``.

    Families in sorted order, so two maps of the same inputs are
    byte-identical. Returns the number of lines (without the header).
    """
    families = sorted(families if families is not None else prep['families'])
    n = 0
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\t'.join(MAP_COLUMNS) + '\n')
        for fam in families:
            for r in family_map_rows(fam, prep['records'][fam]):
                f.write('\t'.join(map(str, r)) + '\n')
                n += 1
    os.replace(tmp, path)
    return n


def read_pair_map(path):
    """``{family: {(row_label, col_label): (LX, LY, file, transpose)}}``."""
    out = {}
    with open(path) as f:
        header = f.readline().rstrip('\n').split('\t')
        if tuple(header) != MAP_COLUMNS:
            raise PairMapError(f'{path}: header {header}')
        for k, line in enumerate(f, start=2):
            fields = line.rstrip('\n').split('\t')
            if len(fields) != len(MAP_COLUMNS):
                raise PairMapError(f'{path}:{k}: {len(fields)} fields')
            fam, row, col, lx, ly, file, tr = fields
            if tr not in ('0', '1'):
                raise PairMapError(f'{path}:{k}: transpose {tr!r}')
            pairs = out.setdefault(fam, {})
            if (row, col) in pairs or (col, row) in pairs:
                raise PairMapError(f'{path}:{k}: pair {row}, {col} repeated')
            pairs[(row, col)] = (int(lx), int(ly), file, int(tr))
    return out


def family_fasta_seqs(fam):
    """``[(name, seq)]`` that MUSCLE must read, in ``fam['names']`` order.

    Decoded from ``fam['int_seqs']`` (the encoding the posteriors are
    indexed by): ``AA_CHARS[k]`` for k < 20, ``X`` for the wildcard 20.
    """
    out = []
    for nm in fam['names']:
        enc = np.asarray(fam['int_seqs'][nm])
        if enc.size and (enc.min() < 0 or enc.max() > 20):
            raise PairMapError(f'{nm}: code outside 0..20')
        out.append((nm, ''.join(AA_CHARS[k] if k < 20 else 'X'
                                for k in enc.tolist())))
    return out


def write_family_fasta(fam, path):
    """``seqs.fasta`` of one family, one line per sequence."""
    with open(path, 'w') as f:
        for nm, s in family_fasta_seqs(fam):
            f.write(f'>{nm}\n{s}\n')


def read_npy_strict(path):
    """``np.load``, after the checks MUSCLE's reader makes: format 1.0 or
    2.0, ``descr '<f4'``, ``fortran_order False``, 2-D, file size exact."""
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        if version not in ((1, 0), (2, 0)):
            raise PairMapError(f'{path}: npy version {version}')
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    if dtype.str != '<f4' or fortran or len(shape) != 2:
        raise PairMapError(f'{path}: dtype {dtype.str}, fortran {fortran}, '
                           f'shape {shape}')
    if os.path.getsize(path) != offset + 4 * shape[0] * shape[1]:
        raise PairMapError(f'{path}: file size does not match shape {shape}')
    return np.load(path)


def load_pairs_from_map(pair_map, post_dir, family, names):
    """The matrices MUSCLE gets for ``family``, as it will read them.

    Returns ``{(i, j): float32 (L_i, L_j)}`` with i < j indexes in
    ``names``. Checks each file's shape against the map line.
    """
    index = {nm: i for i, nm in enumerate(names)}
    out = {}
    for (row, col), (lx, ly, file, tr) in pair_map[family].items():
        i, j = index[row], index[col]
        if i >= j:
            raise PairMapError(f'{family}: row {row} comes after col {col}')
        P = read_npy_strict(os.path.join(post_dir, file))
        if P.shape != ((ly, lx) if tr else (lx, ly)):
            raise PairMapError(f'{family}: {file} has shape {P.shape}, map '
                               f'says {(lx, ly)} transpose {tr}')
        out[(i, j)] = np.ascontiguousarray(P.T if tr else P)
    return out


class MuscleError(RuntimeError):
    """MUSCLE exited non-zero, or its output does not match the input."""


def run_muscle(fasta, out_afa, muscle_bin, post_dir, pair_map, family,
               consiters=2, refineiters=100, threads=1, extra=(), log=None):
    """Align ``fasta`` (``write_family_fasta``'s file) with the given
    posteriors. Output rows are in input order (``-input_order``).

    Returns MUSCLE's stderr; raises ``MuscleError`` if it exits non-zero.
    """
    cmd = [muscle_bin, '-align', fasta, '-output', out_afa,
           '-postdir', post_dir, '-postmap', pair_map,
           '-postfamily', family, '-input_order',
           '-consiters', consiters, '-refineiters', refineiters,
           '-threads', threads, *extra]
    if log is not None:
        cmd += ['-log', log]
    cmd = [str(c) for c in cmd]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise MuscleError(f'{family}: {" ".join(cmd)} exited '
                          f'{p.returncode}:\n{p.stderr[-2000:]}')
    return p.stderr


def read_afa(path):
    """``[(label, row)]`` of an aligned FASTA (rows may span lines)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                out.append([line[1:], []])
            elif out:
                out[-1][1].append(line.strip())
            elif line.strip():
                raise MuscleError(f'{path}: text before the first header')
    return [(lab, ''.join(parts)) for lab, parts in out]


def read_msa(out_afa, fam):
    """MUSCLE's MSA of one family as ``{name: row}``, in ``names`` order.

    Checks that the labels are ``fam['names']``, each once, that the rows
    have one width, that each row without gaps is the sequence of
    ``seqs.fasta``, and the reference ``check_msa`` against ``raw_seqs``.
    The order is not checked: with ``-input_order`` MUSCLE still puts a
    duplicate sequence right after its first copy (InsertDupes runs after
    SortMSA).
    """
    from run_fsa import check_msa
    rows = read_afa(out_afa)
    labels = [lab for lab, _ in rows]
    if sorted(labels) != sorted(fam['names']) or len(set(labels)) != len(
            labels):
        raise MuscleError(f'{out_afa}: labels {labels[:5]}... are not the '
                          f'family names {list(fam["names"])[:5]}...')
    want = dict(family_fasta_seqs(fam))
    widths = {len(r) for _, r in rows}
    if len(widths) != 1:
        raise MuscleError(f'{out_afa}: rows have widths {sorted(widths)}')
    for lab, r in rows:
        if r.replace('-', '') != want[lab]:
            raise MuscleError(f'{out_afa}: row {lab} without gaps differs '
                              f'from seqs.fasta')
    by_label = dict(rows)
    msa = {nm: by_label[nm] for nm in fam['names']}
    check_msa(msa, fam['names'], fam['raw_seqs'])
    return msa
