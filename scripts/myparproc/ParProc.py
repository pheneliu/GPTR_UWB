# User defined functions

# To find number of cores
import os

# To have some delay between updates of progress
import time

# Multi processing
import multiprocessing

# Parallel processing
class ParProc:
    
    def __init__(self, max_proc, proc_id=None):

        self.jobs = []
        self.max_proc = min([os.cpu_count(), max_proc])
        self.proc_id = proc_id
        self.manager = multiprocessing.Manager()
        self.output_queue = self.manager.list([None]*self.max_proc)

    # Add a paraLvxel process that returns some result 
    def addprocwithoutput(self, func, data, idx):
                
        # Add the process
        proc = multiprocessing.Process( target=func, args=(data, idx, self.output_queue))
        self.jobs.append(proc)
        proc.start()

        # Wait until number of jobs is below the threshold before return
        while len(self.jobs) >= self.max_proc:
        
            time.sleep(5)
            
            # Check for finished processes
            finished_proc = []
            for proc in self.jobs:
                if not proc.is_alive():
                    finished_proc.append(proc)
            
            # Remove finished proccesses
            for finproc in finished_proc:
                if finproc in self.jobs:
                    self.jobs.remove(finproc)
                
    # Add a paraLvxel process that does not return any result
    def addproc(self, func, data):

        # Spawn a process to handle the bag
        proc = multiprocessing.Process( target=func, args=( data, ))
        self.jobs.append(proc)
        proc.start()

        # Wait until number of jobs is below the threshold before return
        while len(self.jobs) >= self.max_proc:
            
            time.sleep(5)
            
            # Check for finished processes
            finished_proc = []
            for proc in self.jobs:
                if not proc.is_alive():
                    finished_proc.append(proc)
            
            # Remove finished proccesses
            for finproc in finished_proc:
                if finproc in self.jobs:
                    self.jobs.remove(finproc)

    def join(self):

        # Wait until number of jobs is 0
        while len(self.jobs) > 0:
            
            time.sleep(5)
            
            # Check for finished processes
            finished_proc = []
            for proc in self.jobs:
                if not proc.is_alive():
                    finished_proc.append(proc)
            
            # Remove finished proccesses
            for finproc in finished_proc:
                if finproc in self.jobs:
                    self.jobs.remove(finproc)
            
            # Notify the number of remaining jobs
            print(f"Jobs remained: {len(self.jobs)}")

# To calculate floor and ceil
import math

# To output notifcations that can be captured by jupyter kernel
import logging

if __name__ == "__main__":
    print("This is the myparproc.py file.")
