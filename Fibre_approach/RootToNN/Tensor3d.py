import numpy as np

class Tensor3d:
    
    # Implementation of a 3D tensor
    
    def __init__(self, i, j, k):
        
        # defining number of rows/columns per dimension
        
        self.xdim   = i
        self.ydim   = j
        self.zdim   = k
        self.tensor = np.array([np.array([np.zeros(k) for _ in range(j)]) for __ in range(i)])
        
    def set_value(self, i, j, k, value):
        
        # sets entry(i,j,k) to give value
        
        self.tensor[i][j][k] = value
    
    def give_value(self, i, j, k):
        
        # returns value of entry(i,j,k)
        
        return self.tensor[i][j][k]
        
    def save_to_csv(self, csv_name):
        
        # Saves tensor to .csv file
        
        csv = open(csv_name, "w")
        
        for layer in range(k):
            for ypos in range(j):
                for xpos in range(i):
                    csv.write(str(self.give_value(i,j,k))+';')
                csv.write('\n')
            csv.write('\n')
        csv.close()
    
                    
        
