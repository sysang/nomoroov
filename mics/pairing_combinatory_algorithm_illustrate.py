import itertools


WINDOW_SIZE = 5


def operation_id_factory():
    id = 0

    def operation_id():
        nonlocal id
        id = id + 1
        return '{:04d}'.format(id)

    def reset_opid():
        nonlocal id
        id = 0

    return operation_id, reset_opid


def leaf_id_factory():

    id = 0

    def leaf_id():
        nonlocal id
        id = id + 1
        return '{:02d}'.format(id)

    return leaf_id


generate_opid, reset_opid = operation_id_factory()


def cook_training_data(
        list_of_item,
        generate_leafid,
        operation_id=generate_opid(),
        depth_level=0,
        ended_level=0,
        prev_item=None,
        end_item=None,
        counter=0):
    leaf_id = generate_leafid()
    indentation = '\t' * depth_level
    current_item = None
    current_counter = counter
    print(f'\n\n{indentation}>>> . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . >> {leaf_id} / {operation_id}\n')
    print(f'{indentation}[INITIAL] current_item: {current_item} depth_level: {depth_level} ended_level: {ended_level} prev_item: {prev_item} end_item: {end_item} counter: {counter}')
    for item in list_of_item:
        current_item = item
        current_level = depth_level + 1

        if prev_item is None:
            current_counter = current_counter + 1

        if prev_item is not None:
            print(f'{indentation}[{prev_item}, {item}] *** current_level: {current_level} ended_level: {ended_level} item: {item} end_item: {end_item}')

        if item == end_item:
            print(f'{indentation}[BREAK] current_level: {current_level} ended_level: {ended_level} ** end_item: {end_item} **')
            print(f'{indentation}<< . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . <<< {leaf_id} / {operation_id}\n')
            return ended_level, end_item, current_counter

        if current_level < WINDOW_SIZE and current_level > ended_level:
            it0, it1 = itertools.tee(list_of_item)
            ended_level, end_item, current_counter = cook_training_data(
                    list_of_item=it1,
                    depth_level=current_level,
                    ended_level=ended_level,
                    prev_item=item,
                    end_item=end_item,
                    counter=current_counter + 1,
                    generate_leafid=generate_leafid,
                    operation_id=generate_opid(),
                    )
            print(f'{indentation}[RECURSIVE] current_level: {current_level} ended_level: {ended_level} end_item: {end_item} current_counter: {current_counter}')

        if current_level == WINDOW_SIZE:
            print(f'{indentation}[BREAK] ** current_level: {current_level} ** item: {item}')
            print(f'{indentation}<< . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . <<< {leaf_id} / {operation_id}\n')
            return current_level, item, current_counter

        print(f'{indentation}[CONTINUE] current_item: {current_item} current_level: {current_level} ended_level: {ended_level} prev_item: {prev_item} end_item: {end_item} current_counter: {current_counter}')

        if current_counter % WINDOW_SIZE == 1 and prev_item is None:
            print(f'{indentation}[RESTART] current_item: {current_item} current_level: {current_level} ended_level: {ended_level} prev_item: {prev_item} end_item: {end_item} current_counter: {current_counter}')
            it0, it1 = itertools.tee(list_of_item)
            it2 = itertools.chain(iter([item]), it1)
            _generate_leafid = leaf_id_factory()
            ended_level, end_item, counter = cook_training_data(
                list_of_item=it2,
                depth_level=0,
                ended_level=0,
                prev_item=None,
                end_item=None,
                counter=current_counter - 1,
                generate_leafid=_generate_leafid,
                operation_id=generate_opid(),
            )
            print(f'{indentation}[RECURSIVE] current_level: {current_level} ended_level: {ended_level} end_item: {end_item} current_counter: {current_counter}')

    print(f'{indentation}[RETURN] current_item: {current_item} depth_level: {depth_level} end_item: {end_item} current_counter: {current_counter}')
    print(f'{indentation}<< . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . <<< {leaf_id} / {operation_id}\n')
    return depth_level, end_item, current_counter


test_data = [
    range(1, 5),
    range(1, 6),
    range(1, 7),
    range(1, 8),
    range(1, 9),
    range(1, 10),
    range(1, 11),
    range(1, 12),
    range(1, 17),
    range(1, 20),
]

for list_of_number in test_data:
    print('\n\n\n____ TEST CASE ____', [f'i{num}' for num in list_of_number])
    list_of_item = iter([f'i{num}' for num in list_of_number])
    generate_leafid = leaf_id_factory()
    reset_opid()
    cook_training_data(list_of_item=list_of_item, generate_leafid=generate_leafid, operation_id=generate_opid())
