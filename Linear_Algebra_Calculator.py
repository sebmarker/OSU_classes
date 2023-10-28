import numpy as np
from fraction import Fraction
import replit


##DIM: Dimension, DET: Determinate


############################################################################
 #Get rows, columns, matrix (In Fractions)
def get_matrix(calc_type):
  # get number of rows and columns for matrix
  replit.clear()
  only_rows={'LEN'}
  only_columns={'DOT','PROJ'}
  if calc_type=="LCOMB":
    print("")
    print("Type in combination vectors then target vectors in last column")
  elif calc_type=="PROJ":
    print("")
    print("Type in vector 1 (1st) projected onto vector 2 (2nd)")
  if calc_type in only_rows:
    rows=1
    columns=int(input("Number of columns: "))
  elif calc_type in only_columns:
    columns=2
    rows=int(input("Number of rows: "))
  else:
    rows=int(input("Number of rows: "))
    columns=int(input("Number of columns: "))

  # get current values for matrix
  matrix = []
  for i in range(rows):
    row = []
    for j in range (columns):
      value = input(f"Enter number at {i}, {j} or f if fraction: ")
      if value=="f":
        c=int(input("Input numerator: "))
        d=int(input("Input denominator: "))
        value=Fraction(c,d)
      else:
        value=Fraction(value,1)
      row.append(value)
    matrix.append(row)
  matrix = np.array(matrix)
  return rows,columns,matrix
############################################################################
 #Echelon Form Definitions
def echelon_form(matrix,rows,columns):
  #echelon form
  next_column=0
  lastc_zero=0
  for j in range(columns):#goes to each column
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
      if ptal==i or matrix[i][j]==Fraction(1,1): #if current number is piotal 1
        beta=i
      else:
        beta=f_nonzero(matrix,rows,j,k) #gets first non-zero number in column
      if matrix[beta][j]==Fraction(1,1) and i!=j-lastc_zero: #if first number in column is 1 and first value in current [row,column] isn't zero
        co=1
      else: #first number in column isn't 1
        if matrix[i][j]==0:
          matrix=swap(matrix,matrix[0],matrix[beta],0,beta)
        if matrix[i][j]!=Fraction(0,1):
          scale_factor=matrix[i][j] #scale factor to get row zero=1
        else:
          scale_factor=Fraction(1,1)
        scale_factor=Fraction(scale_factor.denominator,scale_factor.numerator)
        matrix=scale(matrix,i,scale_factor)
        co=1
      fone=first_one(matrix,j,rows,lastc_zero)
      if co==1 and matrix[i][j]!=Fraction(0,1) and i!=fone: #first number in column is one, then combine stuff
        matrix=combine(matrix,i,j,fone)    
      i+=1
  return matrix

  #reduced echelon form
  for j in range(columns-1,-1,-1):
    for i in range(rows-1,-1,-1):
      fone=fones(matrix,rows,j)
      if matrix[fone][j]==Fraction(1,1) and fone!=i and matrix[i][j]!=Fraction(0,1) and i!=rows-1:
        matrix=combine(matrix,i,j,fone)
  return matrix

def swap(matrix,rowtop,rowbottom,top,bottom): #swaps two rows
  matrix=list(matrix)
  matrix[top],matrix[bottom]=matrix[bottom],matrix[top]
  matrix=np.array(matrix)
  return (matrix)
  
def scale(matrix,row,factor): #multipies a row by scale factor
  matrix=np.array(matrix)
  matrix[row]=matrix[row]*factor
  return matrix

#takes combines one row with another, only changing one of them
def combine(matrix,row_change,j,row_same): 
  scale_factor=matrix[row_change][j]
  matrix[row_change]=matrix[row_change]-(matrix[row_same]*scale_factor)
  return(matrix)

#finds first one in current column starting from top
def first_one(matrix,column_number_current,rows,lastc_zero):
  a=column_number_current-lastc_zero
  while a<=rows-1:
    if matrix[a][column_number_current]==Fraction(1,1):
      return a
    else:
      a+=1  

#finds pivotal 1, where there are all 0's below and left of it
def piotal(matrix,i,j,rows,columns,lastc_zero):
  if i==rows-1 and i<=j-lastc_zero: #last row only checks to make sure all 0's left of it
    for r in range(columns):
      if matrix[i][r]==Fraction(0,1):
        r+=1
      elif matrix[i][r]==Fraction(1,1):
        return i
      else:
        return -1       
  else: #not last row checks all the rows below it
    for r in range(rows):
      if matrix[r][j]==Fraction(1,1):
        return r
      else:
        r+=1 
      if r==rows-1:
        return -1

def fones(matrix,rows,j): #finds first one starting from bottom
  k=rows-1
  while k>=0:
    if matrix[k][j]==Fraction(1,1):
      return k
    elif matrix[k][j]==Fraction(0,1):
      k-=1
    else:
      return -1
  return -1

def column_ones(matrix,u,j,rows): #determines if column is all 0's
  while u<=rows-1:
    if matrix[u][j]==Fraction(0,1): 
      if u==(rows-1): #if last row and everything above it is 0
        return 1
    else:
      return 0
    u+=1

def f_nonzero(matrix,rows,j,k): #find first non-zero value in column
  while k<=rows-1: 
    if matrix[k][j]!=Fraction(0,1):
      return k
    k+=1

############################################################################
 #Reduced Echelon Form Definitions
def reduced_echelon_form(matrix,rows,columns):
  matrix=echelon_form(matrix,rows,columns)
  for j in range(columns-1,-1,-1):
    for i in range(rows-1,-1,-1):
      fone=fones(matrix,rows,j)
      if matrix[fone,j]==Fraction(1,1) and fone!=i and matrix[i,j]!=Fraction(0,1) and i!=rows-1:
        matrix=combine(matrix,i,j,fone)
  return matrix
############################################################################
 #Transpose Definitions
def transpose(matrix,number_of_rows,number_of_columns): #switch rows and columns    
  new_matrix=[]
  for i in range(number_of_columns):
    r=[]
    for j in range(number_of_rows):
      r.append(Fraction(0,1))
    new_matrix.append(r)
  for j in range(number_of_columns):
    for i in range(number_of_rows):
      new_matrix[j][i]=matrix[i][j]
  return new_matrix
############################################################################
 #Addition Definitions
def addition_matricies(matrix_a,rows_a,columns_a,matrix_b,rows_b,columns_b):
  if rows_a==rows_b and columns_a==columns_b:
    matrix_c=matrix_b+matrix_a
    return matrix_c
  else:
    return "Error, rows and columns must be same for both matricies"

############################################################################
  #Matrix Multiply Definitions; rows of A dot product columns of B
def multiply_matricies(matrix_a,rows_a,columns_a,matrix_b,rows_b,columns_b):
  if columns_a==rows_b:
    matrix_c=[]
    for i in range(columns_a):
      r=[]
      for j in range(rows_a):
        r.append(Fraction(0,1))
      matrix_c.append(r)
    for i in range(rows_a):
      for j in range(columns_b):
        value=Fraction(0,1)
        for k in range(rows_b):
          value+= (matrix_a[i,k]*matrix_b[k,j])
        matrix_c[i][j]=value
    return matrix_c
  else:
    return "Error, columns of A must equal rows of B"

############################################################################
 #Scalar Multiply Definitions
def scalar_matrix_multiply(matrix,scalar):
  return matrix * scalar

############################################################################
 #Inverse Definitions
def identity_matrix(rows,columns):
  if rows==columns:
    identity_matrix=[]
    for i in range(rows):
      row=[]
      for j in range(columns):
        if i==j:
          value=Fraction(1,1)
        else:
          value=Fraction(0,1)
        row.append(value)
      identity_matrix.append(row)
    identity_matrix=np.array(identity_matrix)
  else:
    return "Error, rows and columns must be equal"
  return identity_matrix
  
def inverse(matrix,rows,columns):
  if rows==columns:
    addition_matrix=identity_matrix(rows,columns)
    matrix=np.c_[matrix,addition_matrix]
    #matrix=echelon_form(matrix,rows,columns)
    matrix=reduced_echelon_form(matrix,rows,columns)
    for j in range(1,columns+1,1):
      matrix=np.delete(matrix,0,1)
    return matrix
  else:
    return "Error, rows and columns must be equal"

############################################################################
 #Dot Product Definitions; add the multiplication of each element in matrix
def dot_product(num,matrix_1,rows_1,column_1,rows_2,column_2,matrix_2):
  dot_prod=Fraction(0,1)
  if rows_1!=rows_2:
    return "Columns not of equal length"
  else:
    if num==1:
      for i in range(rows_1):
        dot_prod += matrix_1[i][column_1]*matrix_1[i][column_2]
    elif num==2:
      for j in range(0,rows_1,1):
        dot_prod += matrix_1[j][0] * matrix_2[j][column_2]
  return dot_prod

############################################################################
 #Orthogonal Defintions
def orthongonal_vectors(matrix,rows,columns):
  for j in range(columns):
    for k in range(1,columns):
      if len(matrix[:][k])!=len(matrix[:][j]):
         return "Columns not of equal length"
      elif k!=j:
        perpendicular=dot_product(1,matrix,len(matrix[:][j]),j,len(matrix[:][k]),k,0)
        if perpendicular!=Fraction(0,1):
          return "Vectors are not Orthogonal"
      elif k==columns-1 and j==columns-1:
        return "Vectors are Orthogonal"

############################################################################
 #Projection Defintions
def projection(matrix,rows_1,column_1,rows_2,column_2):
  if rows_1!=rows_2:
    return "Columns not of equal length"
  else:
    n=dot_product(1,matrix,rows_1,column_1,rows_2,column_2,0)
    d=length_vector(matrix[:,column_2],rows_2,"Fproj")
    if d== Fraction(0,1):
      d=1
    scalar=Fraction(n,d)
    return scalar_matrix_multiply(matrix[:,column_2],scalar)

############################################################################
#Vector Length Defintions
def length_vector(matrix,rows,type):
  if type=="Fproj":
    result=Fraction(0,1)
    for j in range(rows):
      term=matrix[j]
      result += term*term
    return result
  else:
    result=0
    for j in range(rows):
      term=matrix[j]
      result += (term.numerator/term.denominator)**2
    return result
############################################################################
#Rank Definitions
def rank(matrix,rows,columns): #number of non-zero rows
  matrix=reduced_echelon_form(matrix,rows,columns)
  rank=0
  for i in range(rows):
    j=0
    while j<columns:
      if matrix[i,j]==Fraction(1,1): #if number in r,c is 1
        for r in range(rows):
          if matrix[r,j]==Fraction(0,1):
            pass
          elif matrix[r,j]==Fraction(1,1) and r==i:
            pass
          else: 
            j+=1
            break
          if r+1==rows: #if last row
            rank +=1
            j=columns
          r+1
      j+=1
  return rank
############################################################################
# Null Space Defintions; all solutions where Ax=0 (free variables)
def remove_zero_rows(matrix,rows,columns):
  i=0
  rows_removed=[]
  while i <rows: #remove zero rows
    j=0
    while j < columns:
      if matrix[i][j]==Fraction(0,1):
        if j==columns-1:
          matrix=np.delete(matrix,i,0)
          rows= rows-1
          rows_removed.append(i)
        j+=1
      else:
        j=columns
        i+=1
  return matrix, rows_removed

def columns_pivotal_ones_rref(matrix,rows,columns): # find columns with pivotal ones in RREF only
  pivotal_ones=[]
  for i in range(rows):
    j=0
    while j<columns:
      if matrix[i,j]==Fraction(1,1): #if number in r,c is 1
        for r in range(rows):
          if matrix[r,j]==Fraction(0,1):
            pass
          elif matrix[r,j]==Fraction(1,1) and r==i:
            pass
          else: 
            j+=1
            break
          if r+1==rows: #if last row
            pivotal_ones.append(j)
            j=columns
          r+1
      j+=1
  return pivotal_ones
def null_space_matrix(matrix,rows,columns):
  #matrix=echelon_form(matrix,rows,columns)
  matrix=reduced_echelon_form(matrix,rows,columns)
  zero_column=[]
  for j in range(columns): #find zero columns
    for i in range(rows):
      for r in range(rows):
        if matrix[r,j]==Fraction(0,1):
          if r==rows-1:
            if j not in zero_column:
              zero_column.append(j)
        else:
          break
  pivotal_ones=columns_pivotal_ones_rref(matrix,rows,columns)  # find columns with pivotal ones
  #get opposite pivotal one columns
  not_pivotal=[]
  for i in range(columns):
    if i not in pivotal_ones:
      if i not in zero_column:
        not_pivotal.append(i)
  matrix,rows_removed=remove_zero_rows(matrix,rows,columns)
  # get vector matrix
  vector_matrix=[]
  for j in not_pivotal:
    r=[]
    i=0
    for k in range(columns-len(zero_column)):
      if k in pivotal_ones:
        r.append(matrix[i,j]*Fraction(-1,1))
        i+=1
      elif k==j:
        r.append(Fraction(1,1))
      else:
        r.append(Fraction(0,1))
    vector_matrix.append(r)
  vector_matrix=transpose(vector_matrix,len(not_pivotal),columns-len(zero_column))
  #convert to x-values
  x_values=[]
  for i in not_pivotal:
    x_values.append("x"+str(i+1))
  return vector_matrix, x_values
############################################################################
#Range functions; all solutions so that Ax=b has a solution (dependent variables)
def range_matrix(matrix,rows,columns):
  addition_matrix=identity_matrix(rows,rows)
  matrix=np.c_[matrix,addition_matrix]  #add identity matrix to end
  #matrix=echelon_form(matrix,rows,columns)
  matrix=reduced_echelon_form(matrix,rows,columns)
  matrix,rows_removed=remove_zero_rows(matrix,rows,columns+rows)
  rows-= len(rows_removed)
  rows_with_restrictions=[]
  i=0
  while i <rows: #remove zero rows
    j=0
    while j < len(matrix[0])-rows:
      if matrix[i][j]==Fraction(0,1):
        if j==len(matrix[i])-rows-1:
          rows_with_restrictions.append(i)
          i+=1
          break
        j+=1
      else:
        j=len(matrix[i])-rows
        i+=1
  #removes all columns but added ones
  if len(rows_with_restrictions)>0: 
    range_matrix=matrix[:,list(range(len(matrix[0])-rows,len(matrix[0])))]
    range_matrix=range_matrix[rows_with_restrictions,:]
  else:
      return ("All values are in the range")
  return range_matrix
############################################################################
#Basis functions
def basis_method_1(matrix,pivotal_ones,rows):
  new_matrix=[]
  for i in range(rows):
    c=[]
    for j in pivotal_ones:
      c.append(matrix[i][j])
    new_matrix.append(c)
  return new_matrix
############################################################################
#Linear combination function; find a target function as a linear combination of vectors
def linear_combination_orthogonal(matrix,rows,columns): 
  matrix=reduced_echelon_form(matrix,rows,columns)
  result=matrix[:,columns-1]
  rank_result=rank(matrix,rows,columns)
  if rank_result == columns:
    return "No linear combination"
  else:
    return result
############################################################################
#Orthogonal basis; creating an orthogonal basis from a given basis
def column_matrix(matrix): #column matrix from vector
  new_matrix=[]
  new_matrix.append(matrix)
  new_matrix=np.array(new_matrix)
  return new_matrix
def create_orthogonal_basis(matrix,rows,columns):
  result_matrix=[]
  result_matrix.append(matrix[:,0])
  result_matrix=np.array(result_matrix)
  result_matrix=transpose(result_matrix,1,rows)
  result_matrix=np.array(result_matrix)
  for j in range(1,columns): #since 1st columns is added above
    for k in range(j):
      row=matrix[:,j]
      row=column_matrix(row)
      row=transpose(row,1,rows)
      row=np.c_[row,result_matrix[:,k]]
      row_row=[]
      row_row.append(row[:,0])
      row_row=np.array(row_row)
      b=transpose(column_matrix(projection(row,rows,0,rows,1)),1,rows)
      b=np.array(b) *-1
      if k==0:
        row_result=b
      else:
        row_result=addition_matricies(row_result,rows,1,b,rows,1)
    row_result=np.array(row_result) *-1
    result_matrix=np.c_[result_matrix,addition_matricies(transpose(column_matrix(matrix[:,j]),1,rows),rows,1,row_result*-1,rows,1)]
  return result_matrix

# Orthonormal Basis
def orthonormal_basis(matrix,rows,columns):
  for j in range(columns):
    scalar=length_vector(matrix[:,j],rows,"Fproj")
    print(scalar)
    if scalar == Fraction(0,1):
      scalar=Fraction(0,1)
    else:
      scalar=Fraction((scalar.denominator)**(1/2),(scalar.numerator)**(1/2))
    print(scalar)
    matrix[:,j]=scalar_matrix_multiply(matrix[:,j],scalar)
  return matrix
############################################################################
print("Welcome to Linear Algebra Calculator!")
# calc_type=""
# repeat=1
# rows=3
# columns=6
# matrix=[[Fraction(1/1), Fraction(0/1), Fraction(0/1), Fraction(-2/1), Fraction(0/1), Fraction(3/1)],[Fraction(0/1), Fraction(0/1), Fraction(1/1), Fraction(-4/1), Fraction(0/1), Fraction(1/1)],[Fraction(0/1), Fraction(0/1), Fraction(0/1), Fraction(0/1), Fraction(1/1), Fraction(2/1)]]
# matrix=np.array(matrix)

# calc_type="
# repeat=1
# rows=3
# columns=6
# matrix=[[Fraction(1/1), Fraction(1/1), Fraction(0/1), Fraction(0/1), Fraction(-1/1), Fraction(1/1)],[Fraction(0/1), Fraction(1/1), Fraction(2/1), Fraction(1/1), Fraction(3/1), Fraction(1/1)],[Fraction(1/1), Fraction(0/1), Fraction(-1/1), Fraction(1/1), Fraction(1/1), Fraction(0/1)]]
# matrix=np.array(matrix)

# calc_type=""
# repeat=1
# rows=3
# columns=6
# matrix=[[Fraction(0,1),Fraction(1,1),Fraction(3,1),Fraction(10,1),Fraction(-7,1),Fraction(-1,2)],[Fraction(0,1),Fraction(0,1),Fraction(1,1),Fraction(9,1),Fraction(4,1),Fraction(3,1)],[Fraction(0,1),Fraction(3,5),Fraction(0,1),Fraction(1,1),Fraction(-1,2),Fraction(2,3)]]
# matrix=np.array(matrix)

# calc_type="N"
# repeat=1
# rows=3
# columns=5
# matrix=[[Fraction(1,1),Fraction(-1,1),Fraction(1,1),Fraction(0,1),Fraction(0,1)],[Fraction(2,1),Fraction(0,1),Fraction(1,1),Fraction(1,1),Fraction(0,1)],[Fraction(0,1),Fraction(1,1),Fraction(-2,1),Fraction(-1,1),Fraction(0,1)]]
# matrix=np.array(matrix)

# calc_type="ORNB"
# repeat=1
# rows=3
# columns=5
# matrix=[[Fraction(1,1),Fraction(-1,1),Fraction(1,1),Fraction(0,1),Fraction(0,1)],[Fraction(2,1),Fraction(0,1),Fraction(1,1),Fraction(1,1),Fraction(0,1)],[Fraction(0,1),Fraction(1,1),Fraction(-2,1),Fraction(-1,1),Fraction(0,1)]]
# matrix=np.array(matrix)

# calc_type="""
# repeat=1
# rows=3
# columns=4
# matrix=[[Fraction(1,1),Fraction(1,1),Fraction(3,1),Fraction(1,1)],[Fraction(1,1),Fraction(2,1),Fraction(-1,1),Fraction(-4,1)],[Fraction(1,1),Fraction(1,1),Fraction(-1,1),Fraction(7,1)]]
# matrix=np.array(matrix)

# calc_type="LCOMB"
# repeat=1
# rows=3
# columns=4
# matrix=[[Fraction(1,1),Fraction(1,1),Fraction(3,1),Fraction(1,1)],[Fraction(-4,1),Fraction(2,1),Fraction(-1,1),Fraction(1,1)],[Fraction(7,1),Fraction(1,1),Fraction(-1,1),Fraction(1,1)]]
# matrix=np.array(matrix)

#Get matrix and matrix calculation
calc_type=0
repeat=0
print("  ")
one_matrix={'REF','RREF','T','SM','I','LEN','r','R','N','NU','BA1','BA2', 'OR','DOT','PROJ','ORB','LCOMB','ORNB','DET'}
two_matrix={'A','MM'}
while repeat==0:
  if calc_type in one_matrix:
    rows,columns,matrix=get_matrix(calc_type)
    repeat=1
  elif calc_type in two_matrix:
    rows_1,columns_1,matrix_1=get_matrix(calc_type)
    rows_2,columns_2,matrix_2=get_matrix(calc_type)
    repeat=1
  else:
    calculations=["REF: Row Echelon Form","RREF: Reduced Row Echelon form", "T: Transpose", "A: Addition", "MM: Matrix Multiplication", "SM: Scalar Multiplication", "I: Inverse", "DOT: Dot product", "OR: Orthogonal", "PROJ: Projection", "LEN: Length", "N: Null Space", "NU: Nullity", "R: Range", "r: rank","BA1: Basis corresponding pivotal method","BA2: Basis transpose method","LCOMB: Linear Combination", "ORB: Orthogonal Basis", "ORNB: Orthonormal Basis (gives large fractions)", "not complete: DET: Determinate"]
    for x in calculations:
      print(x)
    print("")
    calc_type=input("Input calculation type: ")
    print("")

#Calculate and print text
if calc_type== "REF":
  matrix=echelon_form(matrix,rows,columns)
  print("Matrix in Echelon Form:")
elif calc_type=="RREF":
  matrix=reduced_echelon_form(matrix,rows,columns)
  print("Matrix in Reduced Echelon Form:")
elif calc_type=="T":
  matrix=transpose(matrix,rows,columns)
  print("Transposed Matrix:")
elif calc_type=="A":
  matrix=addition_matricies(matrix_1,rows_1,columns_1,matrix_2,rows_2,columns_2)
  print("Matricies Added:")
elif calc_type=="MM":
  matrix=multiply_matricies(matrix_1,rows_1,columns_1,matrix_2,rows_2,columns_2)
  print("Matricies Multiplied:")
elif calc_type=="SM":
  scalar=input("Scalar Value or f if fraction: ")
  if scalar=="f":
    c=int(input("Input numerator: "))
    d=int(input("Input denominator: "))
    scalar=Fraction(c,d)
  else:
    scalar=Fraction(scalar,1)
  matrix=scalar_matrix_multiply(matrix,scalar)
  print("Matrix Scalar Multiplied")
elif calc_type=="I":
  matrix=inverse(matrix,rows,columns)
  print("Inverse Matrix:")
elif calc_type=="DOT":
  matrix=dot_product(1,matrix,rows,0,rows,1,0)
  print("Dot Product:")
elif calc_type=="OR":
  matrix=orthongonal_vectors(matrix,rows,columns)
  print("Orthogonal Vector Space:")
elif calc_type=="PROJ":
  matrix=projection(matrix,rows,0,rows,1)
  print("Projection:")
elif calc_type=="LEN":
  matrix=length_vector(matrix[0],columns,"norm")
  print("Length:")
elif calc_type=="r":
  matrix=rank(matrix,rows,columns)
  print("Rank of matrix:")
elif calc_type=="N":
  matrix,piv_ones=null_space_matrix(matrix,rows,columns)
  print("Null Space of matrix:")
elif calc_type=="NU":
  matrix,piv_ones=null_space_matrix(matrix,rows,columns)
  matrix=len(piv_ones)
  print("Nullity of Matrix:")
elif calc_type=="R":
  matrix=range_matrix(matrix,rows,columns)
  print("Range of matrix:")
elif calc_type=="BA1":
  matrix_copy=np.copy(matrix)
  matrix_new_ech=echelon_form(matrix,rows,columns)
  matrix_new_ech=reduced_echelon_form(matrix_new_ech,rows,columns)
  pivotal_ones=columns_pivotal_ones_rref(matrix_new_ech,rows,columns)
  matrix=basis_method_1(matrix_copy,pivotal_ones,rows)
  print("Basis using method 1:")
elif calc_type=="BA2":
  matrix=transpose(matrix,len(matrix),len(matrix[0]))
  matrix=echelon_form(matrix,len(matrix),len(matrix[0]))
  matrix=reduced_echelon_form(matrix,len(matrix),len(matrix[0]))
  matrix,removed_rows=remove_zero_rows(matrix,len(matrix),len(matrix[0]))
  matrix=transpose(matrix,len(matrix),len(matrix[0]))
  print("Basis using method 2:")
elif calc_type== "LCOMB":
  matrix=linear_combination_orthogonal(matrix,rows,columns)
  print("Linear Combination Coefficients:")
elif calc_type== "ORB":
  matrix=create_orthogonal_basis(matrix,rows,columns)
  print("Orthogonal Basis:")
elif calc_type== "ORNB":
  matrix=create_orthogonal_basis(matrix,rows,columns)
  matrix=orthonormal_basis(matrix,rows,columns)
  print("Orthonormal Basis: ***not correct")
elif calc_type== "DET":
  print("Determinate:")

#print out answer
print_types={str,float,int}
print_type_2={"PROJ"}
if type(matrix) in print_types or calc_type=="DOT": #type(matrix) float, integer, string, or single fraction
  print(matrix)
elif calc_type in print_type_2:
  for x in matrix:
    print(x)
elif calc_type == "LCOMB":
  for i in range(columns-1):
    print(matrix[i]," ","x",i+1)
else:
  if calc_type=="N":
    num=0
    for x in piv_ones:
      print (*x)
  elif calc_type=="R":
    a=[]
    for x in range(1,len(matrix[0])+1):
      a.append("b"+str(x))
    print(*a)
  for x in matrix:
    print(*x)