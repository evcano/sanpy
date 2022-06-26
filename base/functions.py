import math
import numpy as np
import operator
import os


def check_missing_logs(log_path, log_names):
    missing_logs = []

    for log_name in log_names:
        log_file = os.path.join(log_path, '{}.log'.format(log_name))

        if not os.path.isfile(log_file):
            missing_logs.append(log_name)

    return missing_logs


def write_log(log_path, log_name, log):
    log_file = os.path.join(log_path, '{}.log'.format(log_name))

    with open(log_file, 'w') as _file:
        _file.writelines('{}\n'.format(d) for d in log)

    return

def distribute_objects(obj_list, nproc, myrank):
    nobj = len(obj_list)
    obj_per_proc = math.floor(nobj / nproc)

    low_bound = myrank * obj_per_proc

    if myrank == nproc - 1:
        up_bound = nobj
    else:
        up_bound = (myrank + 1) * obj_per_proc

    proc_obj = obj_list[low_bound:up_bound]

    return proc_obj


def query_pairs(project, conditions):
    """
    i.e. conditions = ['20 < baz < 30', '0 < dis < 100']

    ncondtions is the number of conditions to satisfy
    count indicates how many conditions a pair satisfies
    """

    ops = {'<': operator.lt,
           '<=': operator.le,
           '>': operator.gt,
           '>=': operator.ge}

    nconditions = len(conditions)
    count = np.zeros(project.pairs.index.size)

    for condition in conditions:
        tmp_count = np.zeros(project.pairs.index.size)

        condition = condition.split(" ")
        x1, o1, attr, o2, x2 = condition
        x1 = float(x1)
        x2 = float(x2)

        if attr == "dis" or attr == "baz":
            bank_table = project.pairs.copy()

        idx = np.where(np.logical_and(ops[o1](x1, bank_table[attr]),
                                      ops[o2](bank_table[attr], x2)))
        idx = idx[0]

        tmp_count[idx] = 1
        count += tmp_count

    idx = np.where(count == nconditions)
    queried_pairs = list(project.pairs.index[idx].copy())

    return queried_pairs


def query_virtual_source(pairs_list, virtual_source):
    pairs_list = np.array(pairs_list)

    sta1 = np.array([x.split("_")[0] for x in pairs_list])
    sta2 = np.array([x.split("_")[1] for x in pairs_list])

    idx = np.where(np.logical_or(sta1 == virtual_source,
                                 sta2 == virtual_source))
    idx = idx[0]

    queried_pairs = pairs_list[idx]

    return queried_pairs
