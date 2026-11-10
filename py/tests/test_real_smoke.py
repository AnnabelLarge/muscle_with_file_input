"""Test 6 of the plan: MUSCLE with real posteriors on a few small families.

neuralTKF posteriors of sabre-twi and prefab (on /mnt/sdb, or under
``$FSA_REAL_POSTERIORS``; skipped when absent). ``run_muscle.run_dataset``
aligns the 5 smallest families of each dataset and one of median size
(and, for prefab, 3 families whose posteriors are shared with another
family); the MSAs must
be written and pass ``check_msa``.

The MSA SP/TC scores (``sp_tc_score``, core columns, as score_msas.py)
are printed next to those of the FSA progressive MSAs with the same
consistency/refinement settings (``ci2_ri100``), when those results are
present. No identity is expected. The FSA MSAs are rescored here and
must reproduce their ``scores.json``, which checks the scoring.
"""

import json
import os

import pytest

from conftest import MUSCLE_BIN
from muscle_posteriors import FSA_SCORING, read_pair_map

import run_muscle

from dataset_pairs import reference_dir  # noqa: E402
from prepare import DEFAULT_REFERENCES_DIR, prepare_dataset  # noqa: E402
from run_fsa import check_msa  # noqa: E402
from vendored.scoring import parse_fasta, sp_tc_score  # noqa: E402

REAL_ROOT = (os.environ.get('FSA_REAL_POSTERIORS')
             or '/mnt/sdb/annabel/2d_posteriors_bali3_ox_prefab_sabre')
FSA_RESULTS = (os.environ.get('FSA_CI2_RI100_RESULTS')
               or '/mnt/sdb/annabel/updated_decoding_results/ci2_ri100/'
                  'neuralTKF')
DATASETS = {'sabre-twi': ('neuralTKF_sabre', 'SORTED-SABRE-TWI-DEDUP'),
            'prefab': ('neuralTKF_prefab4ref', 'SORTED-PREFAB4REF-DEDUP')}


def pick_families(prep, n_small=5, n_shared=3):
    size = lambda f: (len(prep['records'][f]['names']), f)
    by_size = sorted(prep['families'], key=size)
    small = by_size[:n_small] + [by_size[len(by_size) // 2]]   # + median
    users = {}
    for f in prep['families']:
        for p in prep['records'][f]['provenance']:
            users.setdefault(p['cache_file'], set()).add(f)
    shared = sorted({f for u in users.values() if len(u) > 1 for f in u},
                    key=size)
    return sorted(set(small) | set(shared[:n_shared]))


@pytest.fixture(scope='module', params=sorted(DATASETS))
def real_run(request, muscle_bin, tmp_path_factory):
    dataset = request.param
    sub, set_dir = DATASETS[dataset]
    root = os.path.join(REAL_ROOT, sub)
    if not os.path.isdir(os.path.join(root, set_dir)):
        pytest.skip(f'no real posteriors under {root}')
    out = str(tmp_path_factory.mktemp(dataset))
    families = pick_families(prepare_dataset(dataset, root, out,
                                             n_workers=8))
    summary = run_muscle.run_dataset(dataset, root, out, muscle_bin=MUSCLE_BIN,
                                     families=families, jobs=4)
    return dataset, out, families, summary


def test_msas_written_and_checked(real_run):
    dataset, out, families, summary = real_run
    assert sorted(summary['per_family']) == families
    prep_records = json.load(open(os.path.join(out, 'muscle_run.json')))
    assert prep_records['options'] == {'consiters': 2, 'refineiters': 100,
                                       'threads': 1, 'input_order': True}
    pm = read_pair_map(os.path.join(out, 'pair_map.tsv'))
    assert sorted(pm) == families
    in_dir = os.path.join(DEFAULT_REFERENCES_DIR,
                          'sabre' if dataset == 'sabre-twi' else 'prefab4ref',
                          'in')
    for f in families:
        msa = parse_fasta(os.path.join(out, 'msas', 'muscle', f'{f}.fasta'))
        raw = parse_fasta(os.path.join(in_dir, f))
        check_msa(msa, list(raw), raw)
        log = open(os.path.join(out, 'muscle', f, 'muscle.log')).read()
        assert f'family {f}' in log and 'Posteriors: from' in log


def test_scores_next_to_fsa(real_run):
    dataset, out, families, summary = real_run
    ref_dir = reference_dir(dataset, DEFAULT_REFERENCES_DIR)
    fsa_dir = os.path.join(FSA_RESULTS, dataset)
    fsa_scores = {}
    if os.path.exists(os.path.join(fsa_dir, 'scores.json')):
        fsa_scores = {r['family']: r for r in json.load(
            open(os.path.join(fsa_dir, 'scores.json')))['per_family']}
    lines = [f'\n{dataset}: family  n_seqs  MUSCLE sp/tc   FSA ci2_ri100 sp/tc']
    sums = [0.0, 0.0, 0.0, 0.0]
    for f in families:
        ref = parse_fasta(os.path.join(ref_dir, f))
        msa = parse_fasta(os.path.join(out, 'msas', 'muscle', f'{f}.fasta'))
        sp, tc = sp_tc_score(msa, ref, core_only=True)
        assert 0 <= sp <= 1 and 0 <= tc <= 1
        row = f'  {f:14s} {summary["per_family"][f]["n_seqs"]:3d}   ' \
              f'{sp:.3f} / {tc:.3f}'
        if f in fsa_scores:
            fsa = parse_fasta(os.path.join(fsa_dir, 'msas', 'g1',
                                           f'{f}.fasta'))
            fsp, ftc = sp_tc_score(fsa, ref, core_only=True)
            assert fsp == pytest.approx(fsa_scores[f]['msa_sp_g1'], abs=1e-12)
            assert ftc == pytest.approx(fsa_scores[f]['msa_tc_g1'], abs=1e-12)
            row += f'     {fsp:.3f} / {ftc:.3f}'
            sums = [sums[0] + sp, sums[1] + tc, sums[2] + fsp, sums[3] + ftc]
        lines.append(row)
    if fsa_scores:
        n = len(families)
        lines.append(f'  mean{"":17s}{sums[0] / n:.3f} / {sums[1] / n:.3f}'
                     f'     {sums[2] / n:.3f} / {sums[3] / n:.3f}')
    print('\n'.join(lines))
