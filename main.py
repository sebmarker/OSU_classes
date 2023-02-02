import numpy as np
from fraction import Fraction
############################################################################
 #Get rows, columns, matrix (In Fractions)
def get_matrix():
  # get number of rows and columns for matrix
  rows=int(input("Number of rows: "))
  columns=int(input("Number of columns: "))

  # get current values for matrix
  matrix = []
  for i in range(0, rows, 1):
    row = []
    for j in range (0, columns, 1):
      value = input(f"Enter number at {i}, {j} or f if fraction: ")
      if value=="f":
        c=int(input("Input numerator: "))
        d=int(input("Input denominator: "))
        value=Fraction(c,d)
      else:
        value=Fraction(value,1)
      row.append(value)
    matrix.append(row);
  matrix = np.array(matrix);
  return rows,columns,matrix
############################################################################
 #Echelon Form Definitions
def reduce_echelon_form(matrix,rows,columns):
  #echelon form
  next_column=0
  lastc_zero=0
  for j in range(0,columns,1):#goes to each column
    if next_column==1:
      lastc_zero+=1 #if any column before was a 0's
    i=j-lastc_zero #first row in column is one higher if row before was a 0's
    while i<=rows-1: #goes to each row
      co=0
      if lastc_zero>=1: #if any previous row was a 0's changes starting values for counters
        u=j-lastc_zero
        k=j-lastc_zero
      else:
        u=j
        k=j
      next_column=column_ones(matrix,u,j,rows)
      if next_column==1: #if column is all 0's go to next column
        break
      ptal=piotal(matrix,i,j,rows,columns,lastc_zero) #finds piotal 1
      if ptal==i or matrix[i,j]==Fraction(1,1): #if current number is piotal 1
        beta=i
      else:
        beta=f_nonzero(matrix,rows,j,k) #gets first non-zero number in column
      if matrix[beta,j]==Fraction(1,1) and i!=j-lastc_zero: #if first number in column is 1 and first value in current [row,column] isn't zero
        co=1
      else: #first number in column isn't 1
        if matrix[i,j]==0:
          matrix=swap(matrix,matrix[0],matrix[beta],0,beta)
        if matrix[i,j]!=Fraction(0,1):
          scale_factor=matrix[i,j] #scale factor to get row zero=1
        else:
          scale_factor=Fraction(1,1)
        scale_factor=Fraction(scale_factor.denominator,scale_factor.numerator)
        matrix=scale(matrix,i,scale_factor)
        co=1
      fone=first_one(matrix,j,rows,lastc_zero)
      if co==1 and matrix[i,j]!=Fraction(0,1) and i!=fone: #first number in column is one, then combine stuff
        matrix=combine(matrix,i,j,fone)    
      i+=1

  #reduced echelon form
  for j in range(columns-1,-1,-1):
    for i in range(rows-1,-1,-1):
      fone=fones(matrix,rows,j)
      if matrix[fone,j]==Fraction(1,1) and fone!=i and matrix[i,j]!=Fraction(0,1) and i!=rows-1:
        matrix=combine(matrix,i,j,fone)
  return matrix

def swap(matrix,rowtop,rowbottom,top,bottom): #swaps two rows
  matrix=list(matrix)
  matrix[top],matrix[bottom]=matrix[bottom],matrix[top]
  matrix=np.array(matrix)
  return (matrix)
  
def scale(matrix,row,factor): #multipies a row by scale factor
  matrix[row]=matrix[row]*factor
  return matrix

#takes combines one row with another, only changing one of them
def combine(matrix,row_change,j,row_same): 
  scale_factor=matrix[row_change,j]
  matrix[row_change]=matrix[row_change]-(matrix[row_same]*scale_factor)
  return(matrix)

#finds first one in current column starting from top
def first_one(matrix,column_number_current,rows,lastc_zero):
  a=column_number_current-lastc_zero
  while a<=rows-1:
    if matrix[a,column_number_current]==Fraction(1,1):
      return a
    else:
      a+=1  

#finds piotal 1, where there are all 0's below and left of it
def piotal(matrix,i,j,rows,columns,lastc_zero):
  if i==rows-1 and i<=j-lastc_zero: #last row only checks to make sure all 0's left of it
    for r in range(0,columns,1):
      if matrix[i,r]==Fraction(0,1):
        r+=1
      elif matrix[i,r]==Fraction(1,1):
        return i
      else:
        return -1       
  else: #not last row checks all the rows below it
    for r in range(0,rows,1):
      if matrix[r,j]==Fraction(1,1):
        return r
      else:
        r+=1 
      if r==rows-1:
        return -1

def fones(matrix,rows,j): #finds first one starting from bottom
  k=rows-1
  while k>=0:
    if matrix[k,j]==Fraction(1,1):
      return k
    elif matrix[k,j]==Fraction(0,1):
      k-=1
    else:
      return -1
  return -1

def column_ones(matrix,u,j,rows): #determines if column is all 0's
  while u<=rows-1:
    if matrix[u,j]==Fraction(0,1): 
      if u==(rows-1): #if last row and everything above it is 0
        return 1
    else:
      return 0
    u+=1

def f_nonzero(matrix,rows,j,k): #find first non-zero value in column
  while k<=rows-1: 
    if matrix[k,j]!=Fraction(0,1):
      return k
    k+=1
          
############################################################################
 #Transpose Definitions
def transpose(matrix,number_of_rows,number_of_columns): #switch rows and columns
  new_matrix=np.zeros([number_of_columns,number_of_rows])
  for j in range(0,number_of_columns,1):
    for i in range(0,number_of_rows,1):
      new_matrix[j,i]=((matrix[i,j].numerator)/(matrix[i,j].denominator))
  return new_matrix    
############################################################################

print("Welcome to Linear Algebra Calculator!")
# get calculation type
types=input("RREF or T: ")

# rows=3
# columns=6
# matrix=[[Fraction(1/1), Fraction(0/1), Fraction(0/1), Fraction(-2/1), Fraction(0/1), Fraction(3/1)],[Fraction(0/1), Fraction(0/1), Fraction(1/1), Fraction(-4/1), Fraction(0/1), Fraction(1/1)],[Fraction(0/1), Fraction(0/1), Fraction(0/1), Fraction(0/1), Fraction(1/1), Fraction(2/1)]]
# matrix=np.array(matrix)

# rows=3
# columns=6
# matrix=[[Fraction(1/1), Fraction(1/1), Fraction(0/1), Fraction(0/1), Fraction(-1/1), Fraction(1/1)],[Fraction(0/1), Fraction(1/1), Fraction(2/1), Fraction(1/1), Fraction(3/1), Fraction(1/1)],[Fraction(1/1), Fraction(0/1), Fraction(-1/1), Fraction(1/1), Fraction(1/1), Fraction(0/1)]]
# matrix=np.array(matrix)

# rows=3
# columns=6
# matrix=[[Fraction(0,1),Fraction(1,1),Fraction(3,1),Fraction(10,1),Fraction(-7,1),Fraction(-1,2)],[Fraction(0,1),Fraction(0,1),Fraction(1,1),Fraction(9,1),Fraction(4,1),Fraction(3,1)],[Fraction(0,1),Fraction(3,5),Fraction(0,1),Fraction(1,1),Fraction(-1,2),Fraction(2,3)]]
# matrix=np.array(matrix)

# rows=3
# columns=5
# matrix=[[Fraction(1,1),Fraction(-1,1),Fraction(1,1),Fraction(0,1),Fraction(0,1)],[Fraction(2,1),Fraction(0,1),Fraction(1,1),Fraction(1,1),Fraction(0,1)],[Fraction(0,1),Fraction(1,1),Fraction(-2,1),Fraction(-1,1),Fraction(0,1)]]
# matrix=np.array(matrix)

if types=="RREF":
  rows,columns,matrix=get_matrix()
  matrix=reduce_echelon_form(matrix,rows,columns)
elif types=="T":
  rows,columns,matrix=get_matrix()
  matrix=transpose(matrix,rows,columns)

#print out answer
if types=="RREF":
  print("Matrix in Reduced Echelon Form:")
elif types=="T":
  print("Transposed Matrix:")
for x in matrix:
    print(*x)