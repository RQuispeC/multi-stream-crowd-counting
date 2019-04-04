from  multiprocessing import Process

def _run_parallel_functions(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

def _run_parallel_2(function, train_test_set, out_dir):
    _run_parallel_functions(function(train_test_set[0], out_dir), function(train_test_set[1], out_dir))

def _run_parallel_3(function, train_test_set, out_dir):
    _run_parallel_functions(function(train_test_set[0], out_dir), function(train_test_set[1], out_dir), function(train_test_set[2], out_dir))

def _run_parallel_4(function, train_test_set, out_dir):
    _run_parallel_functions(function(train_test_set[0], out_dir), function(train_test_set[1], out_dir), function(train_test_set[2], out_dir), function(train_test_set[3], out_dir))

def _run_parallel_5(function, train_test_set, out_dir):
    _run_parallel_functions(function(train_test_set[0], out_dir), function(train_test_set[1], out_dir), function(train_test_set[2], out_dir), function(train_test_set[3], out_dir), function(train_test_set[4], out_dir))

__factory = {
    '2': _run_parallel_2,
    '3': _run_parallel_3,
    '4': _run_parallel_4,
    '5': _run_parallel_5,
}

def run_parallel(function, dataset, out_dir, threads):
    if not (dataset.train_test_size >= threads and threads >= 2):
        raise RuntimeError("Maximum number of threads must be at least 2 and at most {}".format(train_test_size))

    for i in range(0, dataset.train_test_size, threads):
        print("running with ", threads)
        print(dataset.train_test_set[i:min(dataset.train_test_size, i + threads)])
        __factory[str(threads)](function, dataset.train_test_set[i:min(dataset.train_test_size, i + threads)], out_dir)
