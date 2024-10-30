# To calculate floor and ceil
import math

# To output notifcations that can be captured by jupyter kernel
import logging

# Progress check
class ProgressCheck:

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    curr_count  = 0
    total_count = 0
    progress    = -1

    def __init__(self, total):
        self.total_count = total

    def updateProgress(self, affix=''):

        # Increment the count
        self.curr_count += 1
        procString = ''
        if self.total_count == 0:
            procString = f'Progress: 100%'
        elif(math.floor(self.curr_count/self.total_count*100) != self.progress or self.curr_count == self.total_count - 1):
            self.progress = math.floor(self.curr_count/self.total_count*100)
            procString = f' Progress: {self.progress:3d}%'

        if procString != '':
            logging.info(affix + procString)
        return procString
    
if __name__ == "__main__":
    print("This is the myprogresscheck.py file.")

    