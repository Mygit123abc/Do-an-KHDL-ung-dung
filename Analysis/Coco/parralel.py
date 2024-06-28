import multiprocessing
from multiprocessing.pool import ThreadPool

class Multitask():
    def task(tasks, func):
        pool = ThreadPool(len(tasks))

        results = []
        for task in tasks:
            results.append(pool.apply_async(func, args=(task,)))

        pool.close()
        pool.join()

        for index in range(len(results)):
            results[index] = results[index].get()

        return results