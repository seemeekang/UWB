# Original Source: https://www.geeksforgeeks.org/find-angles-given-triangle/

# Python3 code to find all three angles  
# of a triangle given coordinate  
# of all three vertices  
import math 
  
# returns square of distance b/w two points  
def lengthSquare(X, Y):  
    xDiff = X[0] - Y[0]  
    yDiff = X[1] - Y[1]  
    return xDiff * xDiff + yDiff * yDiff 
      
def printAngle(A, B, C):  
      
    # Square of lengths be a2, b2, c2  
    a2 = lengthSquare(B, C)  
    b2 = lengthSquare(A, C)  
    c2 = lengthSquare(A, B)  
  
    # length of sides be a, b, c  
    a = math.sqrt(a2)  
    b = math.sqrt(b2)  
    c = math.sqrt(c2)  
  
    # From Cosine law  
    alpha = math.acos((b2 + c2 - a2) /
                         (2 * b * c))  
    betta = math.acos((a2 + c2 - b2) / 
                         (2 * a * c))  
    gamma = math.acos((a2 + b2 - c2) / 
                         (2 * a * b))  
  
    # Converting to degree  
    alpha = alpha * 180 / math.pi  
    betta = betta * 180 / math.pi  
    gamma = gamma * 180 / math.pi 
  
    # printing all the angles  
    # print("alpha : %f" %(alpha))  
    # print("betta : %f" %(betta)) 
    # print("gamma : %f" %(gamma)) 

    angles = [alpha, betta, gamma]
    return angles 


def main():
    A = (0, 0) 
    B = (0, 1)  
    C = (1, 0) 
    alpha, betta, gamma = printAngle(A, B, C)
    
    print("Alpha: ", alpha)  
    print("Betta: ", betta) 
    print("Gamma: ", gamma) 

if __name__ == "__main__":
    main()