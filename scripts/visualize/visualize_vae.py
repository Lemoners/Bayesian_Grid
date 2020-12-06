import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mygrid.MiniGrid.AE.visualize import visualize, test_solve
import datetime

msavefig = os.path.dirname(os.path.abspath(__file__)) + "/pic/visualize_"

if not os.path.exists(os.path.dirname(msavefig)):
    os.mkdir(os.path.dirname(msavefig))

if __name__ == "__main__":
    # time = datetime.datetime.now()
    # time = datetime.datetime.strftime(time,'%m_%d_%H_%M_%S')
    # msavefig += time + ".jpg"
    # # print(msavefig)
    # visualize(1000, msavefig)

    print(test_solve(n=10000))

