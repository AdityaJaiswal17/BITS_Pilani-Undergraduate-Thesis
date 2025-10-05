from ngsolve import *
import numpy as np
from ngsolve import Mesh, VOL
from math import sqrt
import time
import os
import shutil

def intersection(lst1, lst2):
    lst1 = set(lst1)
    lst2 = set(lst2)
    lst3 = list(lst1 & lst2)

    return lst3

def neighbors_elm(el_num):
    v = elements[el_num].vertices

    v0_list = []
    v1_list = []
    v2_list = []
    # loop over all vertices
    curr_el = elements[el_num].nr
    for i in range(3):
        for el in ma[v[i]].elements: # loop over all elements that contain the vertex v

            if i == 0:
                v0_list.append(el.nr)

            elif i == 1:
                v1_list.append(el.nr)
            
            elif i == 2:    
                v2_list.append(el.nr)
        
    v4_0 = intersection(v1_list, v2_list) 
    v4_1 = intersection(v0_list, v2_list)     
    v4_2 = intersection(v0_list, v1_list)

    v4_0.remove(curr_el)
    v4_1.remove(curr_el)
    v4_2.remove(curr_el)

    if len(v4_0) < 1:
        v4_0 = None
    if len(v4_1) < 1:
        v4_1 = None
    if len(v4_2) < 1:
        v4_2 = None

    if v4_0 is not None:
        v4_0 = int(*v4_0)
    if v4_1 is not None:
        v4_1 = int(*v4_1)
    if v4_2 is not None:
        v4_2 = int(*v4_2)

    #gives the opposite element as per our convention
    el_neighbors = [v4_0, v4_1, v4_2]

    return el_neighbors

def neighbors_vert(el_num):

    el_neighbors = neighbor[el_num]
    v4_0_temp = el_neighbors[0]
    v4_1_temp = el_neighbors[1]
    v4_2_temp = el_neighbors[2]

    # Check if None is stored, if so print "None encountered"
    curr_el = elements[el_num].nr

    if v4_0_temp is not None:
        k = v4_0_temp
        v4_0ID = [elm_vert_list[k][0], elm_vert_list[k][1], elm_vert_list[k][2]]
    else:
        v4_0ID = [None, None, None]

    if v4_1_temp is not None:
        k = v4_1_temp
        v4_1ID = [elm_vert_list[k][0], elm_vert_list[k][1], elm_vert_list[k][2]]
    else:
        v4_1ID = [None, None, None]

    if v4_2_temp is not None:
        k = v4_2_temp
        v4_2ID = [elm_vert_list[k][0], elm_vert_list[k][1], elm_vert_list[k][2]]
    else:
        v4_2ID = [None, None, None]
    
    el_vList = [elm_vert_list[curr_el][0], elm_vert_list[curr_el][1], elm_vert_list[curr_el][2]]
    el_vListArr[curr_el] = [elm_vert_list[curr_el][0].nr, elm_vert_list[curr_el][1].nr, elm_vert_list[curr_el][2].nr]

    if v4_0ID is not None :
        v4_0 = set(v4_0ID)^set(el_vList)
        v4_0.remove(el_vList[0])
    else:
        v4_0 = None

    if v4_1ID is not None:
        v4_1 = set(v4_1ID)^set(el_vList)
        v4_1.remove(el_vList[1])
    else:
        v4_1 = None   
    
    if v4_2ID is not None:
        v4_2 = set(v4_2ID)^set(el_vList)
        v4_2.remove(el_vList[2])
    else:
        v4_2 = None

    v4_0 = list(v4_0)
    v4_1 = list(v4_1)
    v4_2 = list(v4_2)

    if len(v4_0) != 1:
            v4_0 = [None]
    if len(v4_1) != 1:
            v4_1 = [None]
    if len(v4_2) != 1:
            v4_2 = [None]

    v_list = ma[el_vList[0]].point, ma[el_vList[1]].point, ma[el_vList[2]].point #gives the vertex coordinates of the element in consideration
    v_list_ID = ma[el_vList[0]].nr, ma[el_vList[1]].nr, ma[el_vList[2]].nr #gives the vertex ID of the element in consideration
    v_listOpp = *v4_0, *v4_1, *v4_2 #gives opposite vertex (neighbor) to each vertex of the element in consideration

    return v_list, v_list_ID, v_listOpp

class vector_operations:

    def vec_norm(vec):
        vec_norm = sqrt(vec[0]**2 + vec[1]**2)
        return vec_norm
    
    def dot_product(vec1, vec2):
        dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
        return dot
    
    def vec_subtract(vec1, vec2):
        vec_sub = [vec1[0]-vec2[0], vec1[1]-vec2[1]]
        return vec_sub

class internal_angles:

    def tri(el_num, i):

        neighbor_vertices = neighbors_vert(el_num)[2]
        element_vertices = neighbors_vert(el_num)[0]
        neighbor_vertices_coords = []
        internal_angles = []

        for j in range(3):
            if neighbor_vertices[j] is not None:
                neighbor_vertices_coords.append(ma[neighbor_vertices[j]].point)
            elif neighbor_vertices[j] is None:
                neighbor_vertices_coords.append(None)

        if i == 0:
            quad_coords = element_vertices[0], element_vertices[1], neighbor_vertices_coords[0], element_vertices[2]
            if quad_coords[2] is not None:
                for vertex in range(4):
                    vec_1 = vector_operations.vec_subtract(quad_coords[(vertex+1)%4], quad_coords[vertex])
                    vec_2 = vector_operations.vec_subtract(quad_coords[(vertex+3)%4], quad_coords[vertex])
                    dot = vector_operations.dot_product(vec_1, vec_2)
                    mag_vec_1 = vector_operations.vec_norm(vec_1)
                    mag_vec_2 = vector_operations.vec_norm(vec_2)
                    cos_theta = dot/(mag_vec_1*mag_vec_2)
                    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                    internal_angles.append(theta_rad)

            elif neighbor_vertices_coords[0] is None:
                internal_angles.append(float('nan'))


        if i == 1:
            quad_coords = element_vertices[0], element_vertices[1], element_vertices[2], neighbor_vertices_coords[1]
            if quad_coords[3] is not None:
                for vertex in range(4):

                    vec_1 = vector_operations.vec_subtract(quad_coords[(vertex+3)%4], quad_coords[vertex])
                    vec_2 = vector_operations.vec_subtract(quad_coords[(vertex+1)%4], quad_coords[vertex])
                    dot = vector_operations.dot_product(vec_1, vec_2)
                    mag_vec_1 = vector_operations.vec_norm(vec_1)
                    mag_vec_2 = vector_operations.vec_norm(vec_2)
                    cos_theta = dot/(mag_vec_1*mag_vec_2)
                    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                    internal_angles.append(theta_rad)

            elif neighbor_vertices_coords[1] is None:
                internal_angles.append(float('nan'))

        if i == 2:
            quad_coords = element_vertices[0], neighbor_vertices_coords[2], element_vertices[1], element_vertices[2]
            if quad_coords[1] is not None:
                for vertex in range(4):

                    vec_1 = vector_operations.vec_subtract(quad_coords[(vertex+1)%4], quad_coords[vertex])
                    vec_2 = vector_operations.vec_subtract(quad_coords[(vertex+3)%4], quad_coords[vertex])
                    dot = vector_operations.dot_product(vec_1, vec_2)
                    mag_vec_1 = vector_operations.vec_norm(vec_1)
                    mag_vec_2 = vector_operations.vec_norm(vec_2)
                    cos_theta = dot/(mag_vec_1*mag_vec_2)
                    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                    internal_angles.append(theta_rad)

            elif neighbor_vertices_coords[2] is None:
                internal_angles.append(float('nan'))    


        return internal_angles


def quality_func_angles(internal_angles):

    if len(internal_angles) != 0:
        quality_temp = max(abs(0.5*np.pi - internal_angles[0]), abs(0.5*np.pi - internal_angles[1]), abs(0.5*np.pi - internal_angles[2]), abs(0.5*np.pi - internal_angles[3]))
        quality = max((1.0 - (2.0/np.pi)*quality_temp), 0)

    elif len(internal_angles) == 0:
        quality = -0.1 # Dummy negative value to indicate that the element is a boundary element so that it is also sorted without the "None" error 

    return quality

def quality_func_aspect_ratio(el_num, i):

    v = elements[el_num].vertices
    v0 = ma[v[0]].point
    v1 = ma[v[1]].point
    v2 = ma[v[2]].point

    if i == 0:
        v_opp = ma[neighbors_vert(el_num)[2][0]].point
        if v_opp is not None:
            sides = [vector_operations.vec_norm(np.subtract(v0, v1)), vector_operations.vec_norm(np.subtract(v1, v_opp)), vector_operations.vec_norm(np.subtract(v_opp, v2)), vector_operations.vec_norm(np.subtract(v2, v0))]
            aspect_ratio = max(sides)/min(sides)
        elif v_opp is None:
            aspect_ratio = 0
    
    if i == 1:
        v_opp = ma[neighbors_vert(el_num)[2][1]].point
        if v_opp is not None:
            sides = [vector_operations.vec_norm(np.subtract(v0, v1)), vector_operations.vec_norm(np.subtract(v1, v2)), vector_operations.vec_norm(np.subtract(v2, v_opp)), vector_operations.vec_norm(np.subtract(v_opp, v0))]
            aspect_ratio = max(sides)/min(sides)
        elif v_opp is None:
            aspect_ratio = 0
    
    if i == 2:
        v_opp = ma[neighbors_vert(el_num)[2][2]].point
        if v_opp is not None:
            sides = [vector_operations.vec_norm(np.subtract(v0, v_opp)), vector_operations.vec_norm(np.subtract(v1, v2)), vector_operations.vec_norm(np.subtract(v2, v0)), vector_operations.vec_norm(np.subtract(v_opp, v1))]
            aspect_ratio = max(sides)/min(sides)
        elif v_opp is None:
            aspect_ratio = 0

    return aspect_ratio

def isBoundaryElement(el_num):

    el_neighbors = neighbors_elm(el_num)

    if el_neighbors[0] is None or el_neighbors[1] is None or el_neighbors[2] is None:
        return True
    else:
        return False
                
                                        
def laplacian_smoothing(times_to_smooth):

    print(f"\nSmoothing the mesh {times_to_smooth} times")
    for smooth_range in range(times_to_smooth): # smooth range is just a placeholder for the number of times to smooth the mesh
        
        ma_recombined = Mesh(recombinedElements)
        n_vert_recombined = ma_recombined.nv # Number of points in the reombined mesh before the conversion of island triangles to complete quads
        n_el_recombined = ma_recombined.ne # Number of elements in the recombined mesh before the conversion of island triangles to complete quads
        
        print(f"--- {smooth_range+1} times smoothed ---")
        v_neighbors = []
        for v in ma_recombined.vertices:
            v_point = ma_recombined[v].point
            v_neighbors_temp = []
            # find out the vertices connected to v
            for edges in ma_recombined[v].edges:
                # print(v.nr, ma_recombined[edges].vertices[0], ma_recombined[edges].vertices[1])
                v_neighbors_temp.append(ma_recombined[edges].vertices[0])
                v_neighbors_temp.append(ma_recombined[edges].vertices[1])
                v_neighbors_temp = list(set(v_neighbors_temp))
                v_neighbors_temp.remove(v)
            v_neighbors.append(v_neighbors_temp)
            # print(v, v_neighbors[v.nr])


        v_neighbors_points = []
        for v in v_neighbors:
            v_neighbors_points_temp = []
            for v_neighbor in v:
                v_neighbors_points_temp.append(ma_recombined[v_neighbor].point)
            v_neighbors_points.append(v_neighbors_points_temp)
        
        file = open("vertices.txt", "w")
        
        # find boundary vertices and print them
        static_vertices = []  
        for el in ma.Elements(VOL):
            neighbor = neighbors_elm(el.nr)
            for i in range(3):
                if neighbor[i] is None:
                    current_vert = neighbors_vert(el.nr)[0][i]
                    opp_vert_to_current_vert = [v for v in neighbors_vert(el.nr)[0] if v!=current_vert]
                    static_vertices.append(opp_vert_to_current_vert)

                quad_internal_angles = internal_angles.tri(el.nr, i)
                
                # preserves the perfect quad shapes
                if not np.isnan(quad_internal_angles[0]):

                    cut_off_criterion = 0
                    for angle in quad_internal_angles:
                        cut_off_criterion += abs(0.5*np.pi - angle)/4.0

                    # with open('quality_check_smooth.txt', 'a') as filehandle:
                    #     print(quad_internal_angles, el.nr, i, cut_off_criterion, file = filehandle)

                    if cut_off_criterion < smoothing_threshold:
                        current_vert = neighbors_vert(el.nr)[0][i]
                        opp_vert_to_current_vert = [v for v in neighbors_vert(el.nr)[0] if v!=current_vert]
                        static_vertices.append(opp_vert_to_current_vert)
                        
        static_vertices = [item for sublist in static_vertices for item in sublist]
        static_vertices = list(set(static_vertices))

        # print(f"static vertices: {len(static_vertices)}")

        for v in ma_recombined.vertices: # not loop around boundary vertices
            # print(v.nr, ma_recombined[v].point)
            if ma_recombined[v].point not in static_vertices:
                v_point = ma_recombined[v].point
                v_point = np.array(v_point)
                v_neighbors_points_temp = v_neighbors_points[v.nr]
                v_neighbors_points_temp = np.array(v_neighbors_points_temp)
                v_neighbors_points_mean = np.mean(v_neighbors_points_temp, axis=0)
                v_point_new =  v_neighbors_points_mean

                # # find elements connected to v_point
                # for el in ma_recombined[v].elements:
                #     v_elements_ID = ma_recombined[el].vertices

                #     if len(v_elements_ID) == 3:
                #         v_elements_point_before = [ma_recombined[v_elements_ID[0]].point, ma_recombined[v_elements_ID[1]].point, ma_recombined[v_elements_ID[2]].point]
                #         v_elements_point_before = [v for v in v_elements_point_before if v != v_point]
                #         vec_a = np.subtract(v_elements_point_before[0], v_point)
                #         vec_b = np.subtract(v_elements_point_before[1], v_point)

                #         vec_c = np.subtract(v_elements_point_before[0], v_point_new)
                #         vec_d = np.subtract(v_elements_point_before[1], v_point_new)
                        
                #         cross_product_before = np.cross(vec_a, vec_b)
                #         cross_product_after = np.cross(vec_c, vec_d)

                #         with open("test.txt", "a") as testfile:
                #             print(v.nr, el.nr, v_elements_ID, v_elements_point_before, v_point_new, file=testfile)

                #         if np.dot(cross_product_before, cross_product_after) < 0:
                #             print(*ma_recombined[v].point, 0.0, file=file)
                #         else:
                #             print(*v_point_new, 0.0, file=file)
                #         # print(v.nr, el.nr, v_elements_point, cross_product)
                #     else:
                #         v_elements_point_before = [ma_recombined[v_elements_ID[0]].point, ma_recombined[v_elements_ID[1]].point, ma_recombined[v_elements_ID[2]].point, ma_recombined[v_elements_ID[3]].point]
                #         # print(v.nr, el.nr, v_elements_point_before, v_point, v_elements_ID)
                #         v_elements_point_before = [v for v in v_elements_point_before if v != v_point]
                #         vec_a = np.subtract(v_elements_point_before[0], v_point)
                #         vec_b = np.subtract(v_elements_point_before[1], v_point)

                #         vec_c = np.subtract(v_elements_point_before[0], v_point_new)
                #         vec_d = np.subtract(v_elements_point_before[1], v_point_new)
                        
                #         cross_product_before = np.cross(vec_a, vec_b)
                #         cross_product_after = np.cross(vec_c, vec_d)

                #         if np.dot(cross_product_before, cross_product_after) < 0:
                #             print(*ma_recombined[v].point, 0.0, file=file)
                #         else:
                #             print(*v_point_new, 0.0, file=file)

                print(*v_point_new, 0.0, file=file)
            else:
                print(*ma_recombined[v].point, 0.0, file=file)
        file.close()

        recombined_mesh_file = open(recombinedElements, "r")
        for count, line in enumerate(recombined_mesh_file):
            if line == "points\n":
                break
        # print(count)
        recombined_mesh_file.close()

        recombined_mesh_file_temp = open(f"temp_{recombinedElements}", "w")
        with open(recombinedElements, "r") as f:
            for i, line in enumerate(f):
                if i == count+2:
                    with open("vertices.txt", "r") as vertices:
                        for vertex in vertices:
                            recombined_mesh_file_temp.write(vertex)
                else:
                    recombined_mesh_file_temp.write(line)
                if i >= count+2:
                    break
        recombined_mesh_file_temp.close()
        recombined_mesh_file_temp = open(f"temp_{recombinedElements}", "a")
        with open(recombinedElements, "r") as f:
            for i, line in enumerate(f):
                if i >= count+2+n_vert_recombined:
                    recombined_mesh_file_temp.write(line)

        recombined_mesh_file_temp.close()

        shutil.move("temp_"+recombinedElements, recombinedElements)
        os.remove("vertices.txt")



class edges_length:

    def edges_length_tri(el_num, i):

        v = elements[el_num].vertices
        v0 = ma[v[0]].point
        v1 = ma[v[1]].point
        v2 = ma[v[2]].point

        if i == 0:
            side_length = vector_operations.vec_norm(np.subtract(v1, v2))
        
        if i == 1:
            side_length = vector_operations.vec_norm(np.subtract(v0, v2))
        
        if i == 2:
            side_length = vector_operations.vec_norm(np.subtract(v1, v0))

        return side_length
    
    def edges_length_quads(el_num,i):
                
        neighbor_vertices = neighbors_vert(el_num)[2]
        element_vertices = neighbors_vert(el_num)[0]
        neighbor_vertices_coords = []
        side_length = []

        for j in range(3):
            if neighbor_vertices[j] is not None:
                neighbor_vertices_coords.append(ma[neighbor_vertices[j]].point)
            elif neighbor_vertices[j] is None:
                neighbor_vertices_coords.append(None)

        if i == 0:
            quad_coords = element_vertices[0], element_vertices[1], neighbor_vertices_coords[0], element_vertices[2]
            if quad_coords[2] is not None:
                for vertex in range(4):
                    side_length.append(vector_operations.vec_norm(np.subtract(quad_coords[(vertex+1)%4], quad_coords[vertex])))

            elif neighbor_vertices_coords[0] is None:
                side_length.append(float('nan'))


        if i == 1:
            quad_coords = element_vertices[0], element_vertices[1], element_vertices[2], neighbor_vertices_coords[1]
            if quad_coords[3] is not None:
                for vertex in range(4):
                    side_length.append(vector_operations.vec_norm(np.subtract(quad_coords[(vertex+3)%4], quad_coords[vertex])))

            elif neighbor_vertices_coords[1] is None:
                side_length.append(float('nan'))

        if i == 2:
            quad_coords = element_vertices[0], neighbor_vertices_coords[2], element_vertices[1], element_vertices[2]
            if quad_coords[1] is not None:
                for vertex in range(4):
                    side_length.append(vector_operations.vec_norm(np.subtract(quad_coords[(vertex+1)%4], quad_coords[vertex])))

            elif neighbor_vertices_coords[2] is None:
                side_length.append(float('nan'))    


        return side_length
    
def boundary_elements_priority(el_num):

    el_neighbors = neighbors_elm(el_num)

    if el_neighbors[0] is None or el_neighbors[1] is None or el_neighbors[2] is None:
        # boundary_elms.append(el_num)
        for bnd in range(len(elm_info)):
            if elm_info[bnd][0] == el_num:
                elm_info[bnd][4] *= 2  # Increase the quality function of boundary elements by a factor of 2
                
                                        
    return elm_info  

# read mesh file
meshFile = "turbulent_flatplate.vol"
recombinedElements = f"recombined_{meshFile}"
weight = 0.8
times_to_smooth = 2
smoothing_threshold = 0.1
combine_boundary_first = True

ma = Mesh(meshFile)

n_vert = ma.nv
n_el = ma.ne
n_edge = ma.nedge
d = np.sqrt(1)  # length of each edge in metric space 
d = d**2

metric_loc = np.zeros((n_el, 3))  # store the metric tensor for each element
edge_qual_heap_temp = []
edge_qual_heap_temp2 = []
edge_qual = [[None for _ in range(3)] for _ in range(n_el)]
edge_qual_temp = []
edge_qual_heap = []
edge_qual_tempLst = [[None for _ in range(3)] for _ in range(n_el)]
edge_qual_heapSorted = []
element_activity = np.ones((n_el,1))

aspect_ratioSorted_el = np.zeros((n_el, 1))
quad_listArr = []   

v4_0List = []
v4_1List = []
v4_2List = []
v_listOpp = []
v_list = []
v_list_ID = []

el_vListArr = np.zeros((n_el, 3))

opp_vertices = [[0,0], [0,0], [0,0]]
opp_vertices_list = [[0] * 3 for i in range(n_el)]
aspectRatioThreshold = 1 # threshold for aspect ratio


elements = np.zeros((n_el,1), dtype = object) # numpy array
elements = [*ma.Elements(VOL)] # using numpy vectorized operations for faster computation (have to use the '*' operator to unpack the elements) - Although, elements is a list of elements, it is stored as a numpy array.
# print(type(elements))        # <class 'list'>

# loop over all elements
for el in ma.Elements(VOL):
    # print(el.nr)
    # print(type(el.nr))      # int
    # print(type(el))         # <class 'ngsolve.comp.Ngs_Element'>
    # print(el)               # <ngsolve.comp.Ngs_Element object at 0x7f8b3b3b3b70>
    
    v = el.vertices           # get the coordinates of the vertices of the element

    # get coordinates of v
    v0 = ma[v[0]].point
    v1 = ma[v[1]].point
    v2 = ma[v[2]].point

    alpha = np.array([v1[0] - v0[0], v2[0] - v1[0], v0[0] - v2[0]])
    beta = np.array([v1[1] - v0[1], v2[1] - v1[1], v0[1] - v2[1]])

    # coefficients of system of 3 linear equations
    a = np.array([alpha[0]**2, alpha[1]**2, alpha[2]**2])
    c = np.array([beta[0]**2, beta[1]**2, beta[2]**2])
    b = np.array([2*alpha[0]*beta[0], 2*alpha[1]*beta[1], 2*alpha[2]*beta[2]])

    # determinant of 3*3 linear system
    det = a[0]*(b[1]*c[2] - b[2]*c[1]) - b[0]*(a[1]*c[2] - a[2]*c[1]) + c[0]*(a[1]*b[2] - a[2]*b[1])

    detx = d*(b[1]*c[2] - b[2]*c[1]) - b[0]*(d*c[2] - d*c[1]) + c[0]*(d*b[2] - d*b[1])
    dety = a[0]*(d*c[2] - d*c[1]) - d*(a[1]*c[2] - a[2]*c[1]) + c[0]*(a[1]*d - a[2]*d)
    detz = a[0]*(b[1]*d - b[2]*d) - b[0]*(a[1]*d - a[2]*d) + d*(a[1]*b[2] - a[2]*b[1])

    # value of unknowns (x , y, z) which are also 3 independent entries of the 2*2 metric tensor [x, y; y, z]
    x = detx/det
    y = dety/det
    z = detz/det
    metric_loc[el.nr] = [x, y, z]    # el.nr gives the element number

    # with open('local_metric.txt', 'a') as filehandle: #writes the local_metric [(x y);(y,z)], appending it each time
    #     print(format(x, '.16e'), "  ", format(y, '.16e'), "  ", format(z, '.16e'), "  ", *v0, *v1, *v2, "  ", el.nr, file=filehandle)

elm_vert_list = np.zeros((n_el,3), dtype = object)   # list
for el in ma.Elements(VOL):
    v = el.vertices
    elm_vert_list[el.nr] = v

edges_list = np.zeros((n_edge,3), dtype = object)
for edges in ma.edges:
    edges_list[edges.nr] = edges

# Finding the metric tensor at each node by doing the average of the metric tensor of the elements that contain the node
implied_metric = np.zeros((n_vert, 3)) # store the metric tensor for each vertex

for v in ma.vertices:        # loop over all vertices
    # print(v.nr)
    vol = 0
    for el in ma[v].elements: # loop over all elements that contain the vertex v

        v0 = ma[elm_vert_list[el.nr][0]].point  # get the coordinates of element el
        v1 = ma[elm_vert_list[el.nr][1]].point
        v2 = ma[elm_vert_list[el.nr][2]].point

        # calculate volume of el
        vol_loc = 1/6*np.abs((v1[0] - v0[0])*(v2[1] - v0[1]) - (v2[0] - v0[0])*(v1[1] - v0[1]))

        # add metric tensor of el to the metric tensor of v
        implied_metric[v.nr] += vol_loc*metric_loc[el.nr]
        vol += vol_loc

    # divide by the volume of the element to get the volume average of metric at node v
    implied_metric[v.nr] /= vol 
    implied_metric[v.nr] = implied_metric[v.nr]
 
    # scale down the implied_metric
    # implied_metric[v.nr] = implied_metric/4

    # with open ('implied_metric.txt', 'a') as filehandle:    
    #     print(implied_metric[v.nr], v.nr, file=filehandle)


neighbor = [[x for x in range(3)] for y in range(n_el)]
for el in elements:
    for i in range(3):
        neighbor[el.nr][i] = neighbors_elm(el.nr)[i]

v_opp = [[x for x in range(3)] for y in range(n_el)]
for el in elements:
    for i in range(3):
        if neighbor[el.nr][i] is not None:
            v_opp[el.nr][i] = ma[neighbors_vert(el.nr)[2][i]].point
        else:
            v_opp[el.nr][i] = None

print(f"Element sorting based on quality function has started")

aspect_ratio_max = 0

start_time = time.time()
for el in ma.Elements(VOL):
    
    neighbor_local = neighbor[el.nr]
    aspect_ratio_local = np.zeros((3,1))

    for i in range(3):

        if neighbor_local[i] is not None:
            aspect_ratio = quality_func_aspect_ratio(el.nr, i) # temporarily storing the aspect ratio quality function
            # quality_func = 2
            aspect_ratio_local[i] = aspect_ratio

        else:
            aspect_ratio = 0
            aspect_ratio_local[i] = aspect_ratio

    aspect_ratio_next_el = max(aspect_ratio_local)
    if aspect_ratio_next_el > aspect_ratio_max:
        aspect_ratio_max = aspect_ratio_next_el 

    print(f"---- Populating aspect ratio list for normalization ---- {(el.nr/n_el)*100} % completed", end= '\r')

end_time_aspect_ratio = time.time()
print(f"Populating aspect ratio list for normalization has finished in {end_time_aspect_ratio-start_time} seconds")

max_aspect_ratio = aspect_ratio_max

elm_info = []

for el in ma.Elements(VOL):
    
    neighbor_local = neighbor[el.nr]
    # print(neighbor_local)

    for i in range(3):

        if neighbor_local[i] is not None:
            edge_vertex = intersection(elements[neighbor_local[i]].vertices, elements[el.nr].vertices)
            common_edge = intersection(ma[edge_vertex[0]].edges, ma[edge_vertex[1]].edges)

            # quality_func = weight*(quality_func_aspect_ratio(el.nr, i)/max_aspect_ratio) + (1-weight)*quality_func_angles(internal_angles.tri(el.nr, i)) # ADJUSTING WEIGHTS HERE
            quality_func = weight*(edges_length.edges_length_tri(el.nr, i)/min(edges_length.edges_length_quads(el.nr, i))) + (1-weight)*quality_func_angles(internal_angles.tri(el.nr, i)) # ADJUSTING WEIGHTS HERE | Not too good for mesh with curved elements/boundaries

            elm_info.append([el.nr, neighbor_local[i], i, edges_list[common_edge[0].nr][0], quality_func, 1]) # Changing this also would require changes below in the recombination part

    print(f"---- Element {el.nr} has been processed ---- {(el.nr/n_el)*100} % completed", end= '\r')


end_time_elm_info = time.time()
print(f"Time taken for elm_info computation: {end_time_elm_info - end_time_aspect_ratio} seconds")

# boundary_elms = []   # ONLY FOR TESTING PURPOSES of first layer recombinations priority
# combine_test = []    # ONLY FOR TESTING PURPOSES of first layer recombinations priority

if combine_boundary_first == True:
    temp_elm_info = []
    for element in enumerate(elm_info):
        if isBoundaryElement(element[1][0]):
            temp_elm_info.append(element[1])

    temp_elm_info.sort(key = lambda x: x[4], reverse = True)

    elm_info.sort(key = lambda x: x[4], reverse = True)

    elm_info = temp_elm_info + elm_info

if combine_boundary_first == False:
    for el in ma.Elements(VOL):
        boundary_elements_priority(el.nr)

    elm_info.sort(key = lambda x: x[4], reverse = True) 

if combine_boundary_first == True:
    print(f"Element sorting based on quality function has finished with boundary elements having higher priority") 
else:
    print(f"Element sorting based on quality function has finished")

# for i in range(len(elm_info)):
#     for j in range(len(boundary_elms)):
#         if boundary_elms[j] == elm_info[i][0]:
#             combine_test.append(elm_info[i])

# with open('quality_check_serial.txt', 'w') as filehandle:
#     for i in range(len(elm_info)):
#         print(elm_info[i], file=filehandle)     

start_time_recombine = time.time()

iterations = 0
iterations_island_triangles = 0
for recombine in enumerate(elm_info):

    curr_el_nr = recombine[1][0]
    edge_activity = recombine[1][5]
    curr_el_neighbor_nr = recombine[1][1]
    elm_edge_nr = recombine[1][2]
    edge_info = recombine[1][3]
    quality = recombine[1][4]

    v0_nr = neighbors_vert(curr_el_nr)[1][0]
    v1_nr = neighbors_vert(curr_el_nr)[1][1]
    v2_nr = neighbors_vert(curr_el_nr)[1][2]
    
    if 1==1:
        if element_activity[curr_el_nr] == 1 and edge_activity == 1:

            with open('tmp_file.vol', 'a') as filehandle:

                if elm_edge_nr == 0:
                    if element_activity[curr_el_neighbor_nr] == 1:

                        angles = internal_angles.tri(curr_el_nr, 0)

                        if np.around(np.sin(angles[1]), decimals=5) == 0 or np.around(np.sin(angles[3]), decimals=5) == 0:
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ", v0_nr+1,"  ", v1_nr+1,"  ", v2_nr+1, file=filehandle)
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ", v1_nr+1,"  ", neighbors_vert(curr_el_nr)[2][0].nr+1,"  ", v2_nr+1, file=filehandle)
                            
                        # elif np.around(np.sin(angles[3]), decimals=5) == 0:
                        #     print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ",  v0_nr+1,"  ", v1_nr+1,"  ", neighbors_vert(curr_el_nr)[2][0].nr+1, file=filehandle)


                        else:
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "4","  ",  v0_nr+1,"  ", v1_nr+1,"  ", neighbors_vert(curr_el_nr)[2][0].nr+1,"  ", v2_nr+1, file=filehandle)
                        
                        element_activity[curr_el_nr] = 0
                        element_activity[curr_el_neighbor_nr] = 0 

                        edges_curr_el = elements[curr_el_nr].edges
                        edges_curr_el_neighbor = elements[curr_el_neighbor_nr].edges

                        iterations += 1
                        percentage = ((2*(iterations+1))/n_el)*100
                        print(f"{iterations} : Element {curr_el_nr} and Element {curr_el_neighbor_nr} recombined ---- {percentage}% completed")

                    
        
                if elm_edge_nr == 1:
                    if element_activity[curr_el_neighbor_nr] == 1:

                        angles = internal_angles.tri(curr_el_nr, 1)

                        if np.around(np.sin(angles[0]), decimals=5) == 0 or np.around(np.sin(angles[2]), decimals=5) == 0:
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ",  v0_nr+1,"  ", v1_nr+1,"  ", v2_nr+1, file=filehandle)
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ",  v0_nr+1,"  ", v2_nr+1,"  ", neighbors_vert(curr_el_nr)[2][1].nr+1, file=filehandle)

                        # elif np.around(np.sin(angles[2]), decimals=5) == 0:
                        #     print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ",  v0_nr+1,"  ", v1_nr+1,"  ", neighbors_vert(curr_el_nr)[2][1].nr+1, file=filehandle)
                            
                        else:
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "4","  ",  v0_nr+1,"  ", v1_nr+1,"  ", v2_nr+1,"  ", neighbors_vert(curr_el_nr)[2][1].nr+1, file=filehandle)

                        
                        element_activity[curr_el_nr] = 0
                        element_activity[curr_el_neighbor_nr] = 0

                        edges_curr_el = elements[curr_el_nr].edges
                        edges_curr_el_neighbor = elements[curr_el_neighbor_nr].edges

                        iterations += 1
                        percentage = ((2*(iterations+1))/n_el)*100
                        print(f"{iterations} : Element {curr_el_nr} and Element {curr_el_neighbor_nr} recombined ---- {percentage}% completed")




                if elm_edge_nr == 2:
                    if element_activity[curr_el_neighbor_nr] == 1:

                        angles = internal_angles.tri(curr_el_nr, 2)

                        if np.around(np.sin(angles[0]), decimals=5) == 0 or np.around(np.sin(angles[2]), decimals=5) == 0:
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ",  v0_nr+1, v1_nr+1, v2_nr+1, file=filehandle)
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ",  v0_nr+1,"  ", neighbors_vert(curr_el_nr)[2][2].nr+1,"  ", v1_nr+1, file=filehandle)    

                        # elif np.around(np.sin(angles[2]), decimals=5) == 0:
                        #     print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ",  v0_nr+1,"  ", neighbors_vert(curr_el_nr)[2][2].nr+1,"  ", v2_nr+1, file=filehandle)
                    
                        else:
                            print("       2","     ","1", "     ", "0", "     ", "0","     ", "4","  ",  v0_nr+1,"  ", neighbors_vert(curr_el_nr)[2][2].nr+1,"  ", v1_nr+1,"  ", v2_nr+1, file=filehandle)

                        element_activity[curr_el_nr] = 0
                        element_activity[curr_el_neighbor_nr] = 0

                        edges_curr_el = elements[curr_el_nr].edges
                        edges_curr_el_neighbor = elements[curr_el_neighbor_nr].edges

                        iterations += 1
                        percentage = ((2*(iterations+1))/n_el)*100
                        print(f"{iterations} : Element {curr_el_nr} and Element {curr_el_neighbor_nr} recombined ---- {percentage}% completed")

            # break
            
            elms_recombined = iterations


isolated_triangles = 0
iterations_island_triangles = iterations
for el in ma.Elements(VOL):

    if element_activity[el.nr] == 1:
        
        isolated_triangles += 1

        v0_nr = neighbors_vert(el.nr)[1][0]
        v1_nr = neighbors_vert(el.nr)[1][1]
        v2_nr = neighbors_vert(el.nr)[1][2]

        with open('tmp_file.vol', 'a') as filehandle:
            print("       2","     ","1", "     ", "0", "     ", "0","     ", "3","  ",  v0_nr+1,"  ", v1_nr+1,"  ", v2_nr+1, file=filehandle)

        percentage = ((elms_recombined*2 + isolated_triangles)/n_el)*100
        print(f"{iterations_island_triangles} : Element {el.nr} is an island triangle ---- {percentage}% completed")
        iterations_island_triangles += 1


with open(recombinedElements, "w") as filehandle:

    mesh_file =  open(meshFile, "r")
    for count, line in enumerate(mesh_file):
        filehandle.write(line)
        if line == "surfaceelements\n":
            filehandle.write(str(elms_recombined+isolated_triangles))
            filehandle.write("\n")
            line_no = count
            break
    mesh_file.close()
    

    next_half_of_destFile = line_no + elms_recombined + isolated_triangles + 1
    next_half_of_sourceFile = line_no + n_el + 1

    tmp_file = open("tmp_file.vol", "r")
    for count, line in enumerate(tmp_file):
        filehandle.write(line)
    tmp_file.close()
    os.remove("tmp_file.vol")
    
    mesh_file =  open(meshFile, "r")
    for count, line in enumerate(mesh_file):
        if count > next_half_of_sourceFile:
            filehandle.write(line)
    mesh_file.close()

filehandle.close()

end_time_recombine = time.time()

print(f"Time taken for aspect ratio list population: {end_time_aspect_ratio - start_time} seconds   |   Time taken for elm_info computation: {end_time_elm_info - end_time_aspect_ratio} seconds   |   Time taken for recombination: {end_time_recombine - start_time_recombine} seconds", flush = True)

laplacian_smoothing(times_to_smooth)