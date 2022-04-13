from turtle import shape
from typing import Dict, Any
from matplotlib.pyplot import setp

import numpy as np

import utils

NDArray = Any


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    
    vertical_seams = np.ndarray.copy(image)
    re_sized2 = np.ndarray.copy(image) 
    #horizontal_seams = np.ndarray.copy(image)
    re_sized = np.ndarray.copy(image) 
    if out_width != image.shape[1]:
        print("size of image:")
        print(image.shape[0])
        print(image.shape[1])
        re_sized = resize_width(image, vertical_seams, out_width, 0, forward_implementation)
        re_sized2 = np.ndarray.copy(re_sized) 
        print("size of resized:")
        print(re_sized.shape[0])
        print(re_sized.shape[1])
        horizontal_seams = np.ndarray.copy(re_sized)
    if out_height != image.shape[0]:
        re_sized=np.rot90(re_sized, k=1)
        print("size of resized after rot90:")
        print(re_sized.shape[0])
        print(re_sized.shape[1])
   
      
        horizontal_seams = np.ndarray.copy(re_sized)  # horiz color will be by the re-sezied mat 
        print("second call to resize_width:")
        re_sized2 = resize_width(re_sized, horizontal_seams, out_height, 1, forward_implementation)
        re_sized2=np.rot90(re_sized2, k=-1)
        horizontal_seams = np.rot90(horizontal_seams, k = -1)
    #raise NotImplementedError('You need to implement this!')
    
    return { 'resized' : re_sized2, 'vertical_seams' : vertical_seams ,'horizontal_seams' : horizontal_seams}


def resize_width(image, colored_seams, k, is_black, forward_implementation):
    #image=image.astype(np.uint8)
    image= PIL.Image.fromarray(image)

    original_length = image.shape[1]
    deleted_seams = []
    i_mat = utils.to_grayscale(image)
    e_mat = utils.get_gradients(image)
    cost_mat=calc_matrix_cots(i_mat)
    print("*********************************")
    print(i_mat.shape)
    print(e_mat.shape)
    help_mat = np.array([[i for i in range(image.shape[1])] for j in range(
        image.shape[0])])
    seam = np.zeros(image.shape[0]).astype(int)
    num_of_col=i_mat.shape[1]
    print("size of help_mat:")
    print(help_mat.shape)
    for i in range(abs(k-image.shape[1])):
        #print("size of imat:")
        #print(i_mat.shape[0])
        #print(i_mat.shape[1])
        m_mat = np.zeros((image.shape[0], image.shape[1]))
        
        calculate_m(e_mat, m_mat, i_mat, forward_implementation,num_of_col,cost_mat)
        
        find_min_seam(m_mat,e_mat,i_mat, forward_implementation, seam,num_of_col,cost_mat)  #change seam (B)
        
        delete_seam(i_mat, seam)  # (C)
        delete_seam(e_mat,seam)
        delete_seam(cost_mat,seam)
        update_cost_mat(cost_mat,i_mat,seam,num_of_col)
        abs_seam=np.zeros(seam.shape[0]).astype(int)
        calc_abs_seam(abs_seam,seam,help_mat)
        delete_seam(help_mat, seam) # (C)
        num_of_col=num_of_col-1
        deleted_seams.append(abs_seam)  # (D)
        
    if original_length < k:
        #duplicate
        dup_mat=duplicate(help_mat,image,num_of_col,k,deleted_seams)
        change_color(colored_seams, deleted_seams, is_black)
        return dup_mat
        
    
    else:
        reduce_mat=reduce(help_mat,image,num_of_col)
        change_color(colored_seams, deleted_seams, is_black)
        return reduce_mat


def calculate_m(e_mat, m_mat, i_mat, forward_implementation,num_of_col, cost_mat):

    m_mat[0]=np.copy(e_mat[0])
    # change "in-place M"
    if(forward_implementation): # with C
        for i in range(1,e_mat.shape[0]):
            for j in range (num_of_col):
                cl,cv,cr=calc_costs(i,j,i_mat,num_of_col,cost_mat)
           
                if j==0:
                     m_mat[i][j]=e_mat[i][j]+min(m_mat[i-1][j]+cv,m_mat[i-1][j+1]+cr)
                elif j==num_of_col-1:
                     m_mat[i][j]=e_mat[i][j]+min(m_mat[i-1][j-1]+cl,m_mat[i-1][j]+cv)
                else:
                    minval= min((m_mat[i-1][j-1]+cl ), (m_mat[i-1][j]+cv) ,( m_mat[i-1][j+1]+cr))
                    m_mat[i][j] = e_mat[i][j] + minval

    else:  # without C
        for i in range(1,e_mat.shape[0]):
            for j in range (num_of_col):
                if j==0:
                     m_mat[i][j]=e_mat[i][j]+min(m_mat[i-1][j],m_mat[i-1][j+1])
                elif j==num_of_col-1:
                     m_mat[i][j]=e_mat[i][j]+min(m_mat[i-1][j-1],m_mat[i-1][j])
                else:
                    m_mat[i][j]=e_mat[i][j]+min(m_mat[i-1][j-1],m_mat[i-1][j],m_mat[i-1][j+1])

        
def calc_costs(i,j,i_mat,num_of_col,cost_mat):
    if(j>=num_of_col):
        print("error2")
    c_ij=cost_mat[i][j]
    if i==0 and j==0:
        cl=255+255
        cv=255
        cr=255+255
    elif i==0 and j== num_of_col-1:
        cl=255+255
        cv=255
        cr=255+255
    elif i==0 and j>0 and j< num_of_col-1:
        cl=c_ij+255
        cv=c_ij
        cr=c_ij+255
       
    elif j==0 and i>0:
        cl=255+255
        cv=255
        cr=255+ abs(i_mat[i-1][j]-i_mat[i][j+1])
    elif j==num_of_col-1 and i>0:
        cl=255+abs(i_mat[i-1][j]-i_mat[i][j-1])
        cv=255
        cr=255+255

    else:
        
        cl=abs(i_mat[i][j+1]-i_mat[i][j-1])+abs(i_mat[i-1][j]-i_mat[i][j-1])
        cv=abs(i_mat[i][j+1]-i_mat[i][j-1])
        cr=abs(i_mat[i][j+1]-i_mat[i][j-1])+abs(i_mat[i-1][j]-i_mat[i][j+1]) 
        
    return cl, cv, cr

def find_min_seam(m_mat, e_mat,i_mat,forward_implementation, seam,num_of_col,cost_mat):
    #find min seam and chane "seam" "in-place"
    
    if forward_implementation:
        cur_row=m_mat.shape[0]-1

        seam[cur_row]=min_arg(m_mat[cur_row],num_of_col)
        cur_col= seam[cur_row]

        for i in range(cur_row,0,-1):  
            cl, cv, cr = calc_costs(i,cur_col,i_mat,num_of_col,cost_mat)
            if cur_col>0 and (abs(m_mat[i][cur_col]-(e_mat[i][cur_col]+m_mat[i-1][cur_col-1]+cl))<0.001):
                cur_col-=1
            elif(abs(m_mat[i][cur_col]-(e_mat[i][cur_col]+m_mat[i-1][cur_col]+cv))<0.001):
                cur_col+=0
            
            elif cur_col<num_of_col and (abs(m_mat[i][cur_col]-(e_mat[i][cur_col]+m_mat[i-1][cur_col+1]+cr))<0.001):
                cur_col+=1
            else:
                
                print("error")
                print(i," ",cur_row," ",cur_col)
                print("cur col=", cur_col, " cur row=", cur_row, " i=" , i)
                print("m_mat[i]cor_col]=", m_mat[i][cur_col])
                print("cl=", cl, " cv=", cv, " cr=", cr)
                print("e_mat[i]cor_col]=", e_mat[i][cur_col])
                print("m_mat[i-1]cor_col]=", m_mat[i-1][cur_col])
                if cur_col-1 >=0:
                    print("m_mat[i-1]cor_col-1]=", m_mat[i-1][cur_col-1])
                if cur_col <num_of_col -1:
                    print("m_mat[i-1]cor_col+1]=", m_mat[i-1][cur_col+1])
            
                cur_col+=1
            seam[i-1]=cur_col

    else:
        cur_row=m_mat.shape[0]-1
        seam[cur_row]=min_arg(m_mat[cur_row],num_of_col)

        cur_col=min_arg(m_mat[cur_row],num_of_col)
        
        for i in range(cur_row,0,-1): 
            #print("i is: ", i)
            #print("cur_col is: ", cur_col)
            #print("m_mat shape is: ", m_mat.shape)
            #print("e_mat shape is: ", e_mat.shape) 
            if(m_mat[i][cur_col]==e_mat[i][cur_col]+m_mat[i-1][cur_col]):
                cur_col+=0
            elif cur_col>0 and (m_mat[i][cur_col]==e_mat[i][cur_col]+m_mat[i-1][cur_col-1]):
                cur_col-=1
            else:
                cur_col+=1

            seam[i-1]=cur_col


def calc_abs_seam(abs_seam,seam,help_mat):
    #print("seam.shape[0]")
    #print(seam.shape[0])
    #print("help_mat.shape")
    #print(help_mat.shape)
    for i in range(seam.shape[0]):
        abs_seam[i]=help_mat[i][seam[i]]
       

def delete_seam(mat, seam):
    # delete indexes seam from mat "in-place"
    for i in range(seam.shape[0]):
        mat[i][seam[i]:-1] = mat[i][seam[i]+1:]

    
        


        #for j in range(seam[i], mat.shape[1]-1):
        #   mat[i][j] = mat[i][j+1]

def change_color(colored_seams, deleted_seams, is_black):
    
    #color the "colored_seams" matrix with black/red according to the seams
    # in "deleted_seams". "In-place:
    if(is_black):
        color=[0,0,0]
    else:
        color=[255,0,0]
    for i in range(len(deleted_seams)):
        for j in range(colored_seams.shape[0]):
            colored_seams[j][deleted_seams[i][j]]=color


def duplicate(help_mat,image,num_of_col,k,deleted_seams):
    print("k:")
    print(k)
    print("num of col:")
    print(num_of_col)
    new_image=np.zeros((help_mat.shape[0],k,3),dtype=np.float16)
    
    deleted_np = np.array(deleted_seams)
    deleted_np=np.rot90(deleted_np,k=-1)
    deleted_np.sort(axis=1)
    print("deleted_np.shape")
    
    print(deleted_np.shape)
    print("image.shape")
    print(image.shape)
    for i in range(image.shape[0]):
        # seam =    [2,4,7,10]
        #seam_index=[0, 1,2,3]
        step_counter = 0 
        cur_seam_index = 0
        for j in range(image.shape[1]):
            
            if(cur_seam_index< deleted_np.shape[1] and deleted_np[i][cur_seam_index]==j):
                new_image[i][step_counter]=image[i][j]
                step_counter+=1
                new_image[i][step_counter]=image[i][j]
                step_counter+=1
                cur_seam_index+=1
            else:
                new_image[i][step_counter]=image[i][j]
                step_counter+=1
    return new_image

    

def reduce(help_mat,image,num_of_col):
    new_mat=np.zeros((help_mat.shape[0],num_of_col,3),dtype=np.float32)
    for i in range (help_mat.shape[0]):
        for j in range (num_of_col):
            new_mat[i][j]=image[i][help_mat[i][j]]
    return new_mat
def min_arg(arr, num_of_vals):
    min_val=arr[0]
    index=0
    for i in range(num_of_vals):
        if(min_val>arr[i]):
            min_val=arr[i]
            index=i
    return index
def calc_matrix_cots(i_mat):
    left = np.roll(i_mat, shift = 1, axis=1)
    left[:,0] = 0 
    right = np.roll(i_mat, shift = -1, axis=1)
    right[:, -1] = 0 
    
    Output = np.abs(left-right)

    return Output
def update_cost_mat(cost_mat,i_mat,seam,num_of_col):
    for i in range(i_mat.shape[0]):
        if(seam[i]==0):
            cost_mat[i][seam[i]]=0
        elif(seam[i]==1):
            cost_mat[i][seam[i]]=abs(i_mat[i][seam[i]-1]-i_mat[i][seam[i]]+1)
        elif(seam[i]==num_of_col-2):
            cost_mat[i][seam[i]-1]=abs(i_mat[i][seam[i]-2]-i_mat[i][seam[i]])
        elif seam[i]==num_of_col-1:
            cost_mat[i][seam[i]-1]=0
        else:

            cost_mat[i][seam[i]]=abs(i_mat[i][seam[i]-1]-i_mat[i][seam[i]]+1)
            cost_mat[i][seam[i]-1]=abs(i_mat[i][seam[i]-2]-i_mat[i][seam[i]])



        


