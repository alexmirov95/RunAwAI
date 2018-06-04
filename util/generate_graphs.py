import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    data_avg = []
    data_top = []
    
    with open(sys.argv[1], 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        for row in reader:
            data_avg.append(500 - float(row[1].strip()))
            data_top.append(500 - float(row[2].strip()))

        csvfile.close()

    plt.plot(data_avg)
    plt.plot(data_top)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness\nTop Fitness')
    plt.ylim(0, 300);
    plt.show()


