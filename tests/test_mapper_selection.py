from tools.config.mapper_selection import return_mapper_list


def test_curated1_contains_separate_mapper_entries():
    mapper_list = return_mapper_list('curated1')

    assert 'puds' in mapper_list
    assert 'Moriik' in mapper_list
    assert 'pudsMoriik' not in mapper_list
    assert mapper_list.count('puds') == 1
    assert mapper_list.count('Moriik') == 1
