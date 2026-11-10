"""Test 1 of the plan: built-in MUSCLE is deterministic.

The same input run twice must give the same .afa, with 1 thread and with
8. The identity tests of -postout / -postdir rely on this.
"""

import pytest

from conftest import muscle, read_bytes

FAMILIES = ['n2', 'n3', 'n5dup', 'n12']


@pytest.mark.parametrize('family', FAMILIES)
def test_same_output_twice_and_any_thread_count(muscle_bin, fasta_dir,
                                                tmp_path, family):
    fa = fasta_dir / f'{family}.fa'
    outs = []
    for threads in (1, 1, 8, 8):
        out = tmp_path / f'{family}.{threads}.{len(outs)}.afa'
        muscle('-align', fa, '-output', out, '-threads', threads)
        outs.append(read_bytes(out))
    assert len(outs[0]) > 0
    assert all(o == outs[0] for o in outs), family


@pytest.mark.parametrize('family', ['n5dup', 'n12'])
@pytest.mark.parametrize('opts', [('-consiters', 0), ('-refineiters', 0),
                                  ('-perm', 'abc'), ('-input_order',)])
def test_same_output_twice_with_options(muscle_bin, fasta_dir, tmp_path,
                                        family, opts):
    fa = fasta_dir / f'{family}.fa'
    outs = []
    for threads in (1, 8):
        out = tmp_path / f'{family}.{threads}.afa'
        muscle('-align', fa, '-output', out, '-threads', threads, *opts)
        outs.append(read_bytes(out))
    assert outs[0] == outs[1]
