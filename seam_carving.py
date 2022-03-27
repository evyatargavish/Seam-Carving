from typing import Dict, Any

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
    horizontal_seams = np.ndarray.copy(image)

    if out_width != image.shape[1]:
        resize_width(image, vertical_seams, out_width, 0, forward_implementation)
    if out_height != image.shape[0]:
        np.rot90(image, k=1)
        resize_width(image, horizontal_seams, out_height, 1, forward_implementation)
        np.rot90(image, k=-1)
    raise NotImplementedError('You need to implement this!')
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}


def resize_width(image, colored_seams, k, is_black, forward_implementation):
    original_length = image.shape[1]
    deleted_seams = []
    i_mat = utils.to_grayscale(image)
    e_mat = utils.get_gradients(image)
    help_mat = np.array([[i for i in range(image.shape[0])] for j in range(
        image.shape[1])])
    seam = np.zeros(image.shape[0])
    for i in range(k):
        m_mat = np.zeros((image.shape[0], image.shape[1]))
        if forward_implementation: # calc M with C
            calculate_m(e_mat, m_mat, i_mat, forward_implementation)
        else: # calc M without C
            calculate_m(e_mat, m_mat, i_mat, forward_implementation)
        find_min_seam(m_mat, forward_implementation, seam)  #change seam (B)
        delete_seam(i_mat, seam)  # (C)
        delete_seam(help_mat, seam) # (C)
        deleted_seams.append(seam)  # (D)
    if original_length < k:
        #duplicate
        pass
    else:
        #reduce
        pass

    change_color(colored_seams, help_mat, is_black)


def calculate_m(e_mat, m_mat, i_mat, forward_implementation):
    # change "in-place M"
    if(forward_implementation): # with C
        pass
    else:  # without C
        pass

def find_min_seam(m_mat, forward_implementation, seam):
    #find min seam and chane "seam" "in-place"
    pass

def delete_seam(mat, seam):
    # delete indexes seam from mat "in-place"
    pass

def change_color(colored_seams, help_mat, is_black):
    #color the "colored_seams" matrix with black/red according to the seams
    # in "help_mat". "In-place:
    pass

def duplicate():
    pass

def reduce():
    pass