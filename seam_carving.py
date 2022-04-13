from asyncio.windows_events import NULL
import re
from turtle import shape
from typing import Dict, Any
from xmlrpc.client import MAXINT
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
    # in case there is no change in both of the sizes:
    if(out_width == image.shape[1] and out_height == image.shape[0]):
        return { 'resized' : image, 'vertical_seams' : image ,'horizontal_seams' : image}


    vertical_seams = np.ndarray.copy(image) #copy of the image with vertical colored seams
    
    #duplicate the images
    re_sized = np.ndarray.copy(image) 
    re_sized2 = np.ndarray.copy(image) 
    

    # change the width:
    if out_width != image.shape[1]:
        re_sized = resize_width(image, vertical_seams, out_width, 0, forward_implementation)
        re_sized2 = np.ndarray.copy(re_sized) 
        horizontal_seams = np.ndarray.copy(re_sized) #copy of the image with horizonal colored seams
    # change the hight
    if out_height != image.shape[0]:
        # rotate the image and repeat the process:
        re_sized=np.rot90(re_sized, k=1)
        horizontal_seams = np.ndarray.copy(re_sized)  # horiz color will be by the re-sezied mat 
        re_sized2 = resize_width(re_sized, horizontal_seams, out_height, 1, forward_implementation)

        #rotate back the image:
        re_sized2=np.rot90(re_sized2, k=-1)
        horizontal_seams = np.rot90(horizontal_seams, k = -1)
    
    
    return { 'resized' : re_sized2, 'vertical_seams' : vertical_seams ,'horizontal_seams' : horizontal_seams}


def resize_width(image, colored_seams, k, is_black, forward_implementation):

    original_length = image.shape[1]
    deleted_seams = [] #save the deleted seams by their original indexes
    i_mat = utils.to_grayscale(image) #gray scale matrix
    e_mat = utils.get_gradients(image) #energy matrix
    cost_mat=calc_matrix_cots(i_mat,i_mat.shape[1]) #calc abs(i_mat((𝑖, 𝑗 + 1) − i_mat((𝑖, 𝑗 − 1))
    
    help_mat = np.array([[i for i in range(image.shape[1])] for j in range(
        image.shape[0])]) # the original idexes (per row) of the image that are still exists
    seam = np.zeros(image.shape[0]).astype(int) #save the current seam (by the current indices- not by the absolute)
    num_of_col=i_mat.shape[1] #save the current right-boundery of the image
  
    for i in range(abs(k-image.shape[1])): # k iterartions to find and delete a seam
        #Use dynamic programming to find the optimal vertical seam by calculating the cost matrix m_mat
        m_mat = np.zeros((image.shape[0], image.shape[1])) 
        calculate_m(e_mat, m_mat, i_mat, forward_implementation,num_of_col,cost_mat)

        #Find the actual seam by finding the smallest cost in the bottom row, then start going up on a path of minimal costs.
        find_min_seam(m_mat,e_mat,i_mat, forward_implementation, seam,num_of_col,cost_mat)  #by the current indices
        #delete the min seam from all relevant matrixes
        delete_seam(i_mat, seam) 
        delete_seam(e_mat,seam)
        delete_seam(cost_mat,seam)

        cost_mat=calc_matrix_cots(i_mat,num_of_col-1) # calc abs(i_mat((𝑖, 𝑗 + 1) − i_mat((𝑖, 𝑗 − 1))
        # calc the ablolute indices of the minimal seam
        abs_seam=np.zeros(seam.shape[0]).astype(int) 
        calc_abs_seam(abs_seam,seam,help_mat)
        
        delete_seam(help_mat, seam) 
        num_of_col=num_of_col-1 #update right-boundary
        deleted_seams.append(abs_seam)  # add the absolute seam to the list
    # Expand the image:    
    if original_length < k:
        #duplicate
        dup_mat=duplicate(help_mat,image,num_of_col,k,deleted_seams)
        change_color(colored_seams, deleted_seams, is_black)
        return dup_mat
        
    #Reduce the image
    else:
        reduce_mat=reduce(help_mat,image,num_of_col)
        change_color(colored_seams, deleted_seams, is_black)
        return reduce_mat

#Use dynamic programming to find the optimal vertical seam by calculating the cost matrix m_mat
def calculate_m(e_mat, m_mat, i_mat, forward_implementation,num_of_col, cost_mat):

    m_mat[0]=np.copy(e_mat[0])
    # change "in-place M"
    if(forward_implementation): # with C
        for i in range(1,e_mat.shape[0]):
            cl=np.abs(np.subtract(np.roll(i_mat[i],1), i_mat[i-1] ))
            cl[0]=255
            cr=np.abs(np.subtract(np.roll(i_mat[i],-1) , i_mat[i-1] ))
            cr[num_of_col-1]=255
            vertical=m_mat[i-1]+cost_mat[i]
            cleft=cost_mat[i]+cl
            cright=cost_mat[i]+cr
            m_left=np.roll(m_mat[i-1],1)
            m_left=m_left+cleft
            m_left[0]=MAXINT
            m_right=np.roll(m_mat[i-1],-1)
            m_right=m_right+cright
            m_right[num_of_col-1]=MAXINT
            m_mat[i] = e_mat[i] + np.minimum(vertical,np.minimum(m_left,m_right))



    else:  # without C
        for i in range(1,e_mat.shape[0]):
            
            right=np.roll(m_mat[i-1],1)
            right[0]=MAXINT
            left=np.roll(m_mat[i-1],-1)
            left[num_of_col-1]=MAXINT
            m_mat[i] = e_mat[i] + np.minimum(m_mat[i-1],np.minimum(left,right))
            
# calc abs(i_mat((𝑖, 𝑗 + 1) − i_mat((𝑖, 𝑗 − 1))        
def calc_costs(i,j,i_mat,num_of_col,cost_mat):
   
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

# Find the actual seam by finding the smallest cost in the bottom row,
def find_min_seam(m_mat, e_mat,i_mat,forward_implementation, seam,num_of_col,cost_mat):
    #find min seam and chanעe "seam" "in-place"
    
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
            else:
                cur_col+=1

            seam[i-1]=cur_col

    else:
        cur_row=m_mat.shape[0]-1
        seam[cur_row]=min_arg(m_mat[cur_row],num_of_col)
        cur_col=min_arg(m_mat[cur_row],num_of_col)
        
        for i in range(cur_row,0,-1): 
            if(m_mat[i][cur_col]==e_mat[i][cur_col]+m_mat[i-1][cur_col]):
                cur_col+=0
            elif cur_col>0 and (m_mat[i][cur_col]==e_mat[i][cur_col]+m_mat[i-1][cur_col-1]):
                cur_col-=1
            else:
                cur_col+=1

            seam[i-1]=cur_col
# calc actual indices of the seam
def calc_abs_seam(abs_seam,seam,help_mat):
    for i in range(seam.shape[0]):
        abs_seam[i]=help_mat[i][seam[i]]
       
# delete the seam from the matrix
def delete_seam(mat, seam):
    # delete indexes seam from mat "in-place"
    for i in range(seam.shape[0]):
        mat[i][seam[i]:-1] = mat[i][seam[i]+1:]

#color the "colored_seams" matrix with black/red according to the seams
def change_color(colored_seams, deleted_seams, is_black):
    
    if(is_black):
        color=[0,0,0]
    else:
        color=[255,0,0]
    for i in range(len(deleted_seams)):
        for j in range(colored_seams.shape[0]):
            colored_seams[j][deleted_seams[i][j]]=color

# in case of extanding the image- use "duplicate" to add relevant seams to the picture
def duplicate(help_mat,image,num_of_col,k,deleted_seams):
    new_image=np.zeros((help_mat.shape[0],k,3),dtype=np.float16)
    deleted_np = np.array(deleted_seams)
    deleted_np=np.rot90(deleted_np,k=-1)
    deleted_np.sort(axis=1)
    for i in range(image.shape[0]):
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
# in case of reducing the image- use "reduce" to remove relevant seams from the picture
def reduce(help_mat,image,num_of_col):
    new_mat=np.zeros((help_mat.shape[0],num_of_col,3),dtype=np.float32)
    for i in range (help_mat.shape[0]):
        for j in range (num_of_col):
            new_mat[i][j]=image[i][help_mat[i][j]]
    return new_mat

#find index of minimum val in the arr (similar to np.argmin but with right boundary)
def min_arg(arr, num_of_vals):
    min_val=arr[0]
    index=0
    for i in range(num_of_vals):
        if(min_val>arr[i]):
            min_val=arr[i]
            index=i
    return index
# calc abs(i_mat((𝑖, 𝑗 + 1) − i_mat((𝑖, 𝑗 − 1))        
def calc_matrix_cots(i_mat,num_of_col):
    left = np.roll(i_mat, shift = 1, axis=1)
    right = np.roll(i_mat, shift = -1, axis=1)
    Output = np.abs(left-right)
    Output[:,0] = 255 
    Output[:, num_of_col-1] =255 

    return Output




