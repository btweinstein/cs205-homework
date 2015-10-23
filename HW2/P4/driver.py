import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import pylab

import filtering
from timer import Timer
import threading

from or_event import OrEvent
from threadpool import ThreadPool

print_lock = threading.Lock()

class Row_Handler():
    def __init__(self, row_num, tmpA, tmpB, num_iterations=10):
        self.i = 0

        self.row_num = row_num

        self.above_handler = None
        self.below_handler = None

        # Run when you are ready to go!
        self.in_loop = threading.Event()
        self.i_updated = threading.Event()

        self.tmpA = tmpA
        self.tmpB = tmpB
        self.num_iterations = num_iterations


    def go(self): #
        print_lock.acquire()
        print 'Go!' , self.row_num, self.i
        print_lock.release()
        while True:
            # Grab in blocks of three.

            above_exists = self.above_handler is not None
            below_exists = self.below_handler is not None

            go_cond_1 = True
            if above_exists:
                go_cond_1 = (self.i == self.above_handler.i)
            go_cond_2 = True
            if below_exists:
                go_cond_2 = (self.i == self.below_handler.i)
            if go_cond_1 and go_cond_2:

                if above_exists and below_exists:
                    self.above_handler.in_loop.wait()

                    cond1 = not self.above_handler.in_loop.is_set()
                    cond2 = not self.in_loop.is_set()
                    cond3 = not self.below_handler.in_loop.is_set()

                    if cond1 and cond2 and cond3:
                        self.above_handler.in_loop.set()
                        self.in_loop.set()
                        self.below_handler.in_loop.set()
                    else:



                if self.above_handler is None:
                    self.above_handler.in_loop.wait()
                    self.above_handler.in_loop.set()
                if self.below_handler is None:
                    self.below_handler.in_loop.wait()
                    self.below_handler.in_loop.set()
                else:
                    OrEvent(self.below_handler.in_loop, self.above_handler.in_loop)

                # Do stuff
                filtering.median_3x3_row(self.tmpA, self.tmpB, self.row_num)
                # Swap tmpA and tmpB; doesn't recreate arrays, just swaps pointers!
                potatoA = self.tmpA
                potatoB = self.tmpB

                self.tmpA = potatoB
                self.tmpB = potatoA

                if self.above_handler is not None:
                    self.above_handler.in_loop.clear()
                self.in_loop.clear() # We are in the loop! Solve deadlocks.
                if self.below_handler is not None:
                    self.below_handler.in_loop.clear()

                # Give other threads a chance to grab locks and update
                self.i += 1
                self.i_updated.set()
                self.i_updated.clear()

                if self.i == self.num_iterations:
                    break

            # Try to go if you neighbors are updated...
            if self.above_handler is None:
                print_lock.acquire()
                print 'Waiting for below handler to update...' , self.row_num, self.i
                print_lock.release()
                self.below_handler.i_updated.wait()
                print_lock.acquire()
                print 'Below handler updated!' , self.row_num, self.i
                print_lock.release()
            elif self.below_handler is None:
                print_lock.acquire()
                print 'Waiting for above handler to update...' , self.row_num, self.i
                print_lock.release()
                self.above_handler.i_updated.wait()
                print_lock.acquire()
                print 'Above handler updated!' , self.row_num, self.i
                print_lock.release()
            else:
                print_lock.acquire()
                print 'Waiting for above or below to update...' , self.row_num, self.i
                print_lock.release()
                OrEvent(self.above_handler.i_updated, self.below_handler.i_updated).wait()
                print_lock.acquire()
                print 'Below or Above handler updated!' , self.row_num, self.i
                print_lock.release()

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Create all row_handlers
    handler_list = []
    for cur_row in range(image.shape[0]):
        handler_list.append(Row_Handler(cur_row, tmpA, tmpB, num_iterations=iterations))

    # Add neighbors
    for i in range(len(handler_list)):
        if i != len(handler_list) - 1:
            handler_list[i].below_handler =handler_list[i + 1]
        if i != 0:
            handler_list[i].above_handler = handler_list[i -1]

    # The handlers must each be updated ten times...if we had a thread for each handler we would be all set...
    # we can probably subdivide the jobs up.

    pool = ThreadPool(num_threads)
    for handler in handler_list:
        pool.add_task(handler.go)

    pool.wait_completion()

    return tmpA

def numpy_median(image, iterations=10):
    ''' filter using numpy '''
    for i in range(iterations):
        padded = np.pad(image, 1, mode='edge')
        stacked = np.dstack((padded[:-2,  :-2], padded[:-2,  1:-1], padded[:-2,  2:],
                             padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
                             padded[2:,   :-2], padded[2:,   1:-1], padded[2:,   2:]))
        image = np.median(stacked, axis=2)

    return image


if __name__ == '__main__':
    input_image = np.load('image.npz')['image'].astype(np.float32)

    pylab.gray()

    pylab.imshow(input_image)
    pylab.title('original image')

    pylab.figure()
    pylab.imshow(input_image[1200:1800, 3000:3500])
    pylab.title('before - zoom')

    # verify correctness
    from_cython = py_median_3x3(input_image, 2, 5)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
