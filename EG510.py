import numpy as np
import math as math

# if user input needs to stored in a list
def get_multiple_input(input_text):
    inputs = []
    while True:
        i = input(input_text)
        try:
            a = int(i)
        except ValueError:
            break
        inputs.append(a)
    return inputs


# if only one value is input
def get_single_input(input_text):
    i = input(input_text)
    while True:
        try:
            i = int(i)
            break
        except ValueError:
            break
    return i

def get_f_p_init_N(t_list, strands_arr, p_force): #strand arrangement, prestress force
    N_array = []
    for i in range(0, len(t_list)):
        indiv_N = 0
        for j in range(0, len(strands_arr)):
            indiv_N += strands_arr[j] * p_force
        N_array.append(indiv_N)
    return N_array

def get_f_p_init_M(t_list, strands_arr, p_force, y_p_list): #strand arrangement, prestress force
    M_array = []
    for i in range(0, len(t_list)):
        indiv_M = 0
        for j in range(0, len(strands_arr)):
            indiv_M += strands_arr[j] * p_force * y_p_list[j]
        M_array.append(indiv_M)
    return M_array

def get_f_p_rel(fpi_n, fpi_m, phi_p_steel_array):
    N_array = []
    M_array = []
    for i in range(0,len(t)):
        N_array.append(fpi_n[i] * phi_p_steel_array[i])
        M_array.append(fpi_m[i] * phi_p_steel_array[i])
    rel = np.array([N_array, M_array], dtype=np.float64)
    return rel

def get_r_b_steel(rb_input):
    rb_type = {0: (2 / 100), 1: (2.5 / 100), 2: (4 / 100)} #create a dictionary for values
    return rb_type[rb_input]

def get_k4_steel(t_list):
    k4s_total = []
    for i in range(0,len(t_list)):
        k4s = math.log10(5.4 * math.pow(t_list[i], 1/6 ))
        k4s_total.append(k4s)
    return k4s_total

def get_k5_steel(break_str, jack_str, rb_steel):
    gamma = jack_str / break_str
    if gamma < 0.4:
        k5_i = 0
    elif 0.4 <= gamma <= 0.7:
        k5_i = (10 * gamma - 4) / 3
    else:
        if rb_steel == 0.02 or 0.025:
            k5_i = 5 * gamma - 2.5 #low relaxation wire or low relaxation strand
        else:
            k5_i = (50 * gamma) / 6 #Alloy Steel bars
    return k5_i

def get_k6_steel(input_temp):
    temp = 20
    i = input_temp
    if i != "":
        temp = int(i)
    return temp / 20

def get_E_c(f_cmi_input, density_conc):
    #f_cmi_input = get_single_input("Enter mean in-situ compressive strength (MPa): ")
    if f_cmi_input <= 40:
        E_c = math.pow(density_conc, 1.5) * 0.043 * math.sqrt(f_cmi_input)
    else:
        E_c = math.pow(density_conc, 1.5) * (0.024 * math.sqrt(f_cmi_input) + 0.12)
    return E_c

def get_E_c_tao_2_t(t2_array, t_slab_pour):
    array = []
    for i in range(0, len(t2_array)):
        output = t2_array[i] - t_slab_pour
        if output == 0:
            output = 1
        else:
            output = output
        array.append(output)
    return array

def get_s(s_key):
    while True:
        if s_key == 0:
            s = 0.38
            break
        elif s_key == 1:
            s = 0.25
            break
        else:
            print("Enter 0 or 1")
    return s

def get_E_c_tao(t_list, s_value, E_c_value):
    E_c_array = []
    for i in range(0, len(t_list)):
        E_c_array.append(math.pow(math.exp(s_value * (1 - math.sqrt(28 / t_list[i]))), 0.5) * E_c_value)
    return E_c_array

def get_th(area, ue):
    return 2*area / ue

def get_alpha2(area, ue):
    th = 2 * area / ue
    return 1 + 1.12 * math.exp(-0.008 * th)

def get_k2_value(area, ue, t_list, tao_list):
    alpha2_value = get_alpha2(area, ue)
    th_value = get_th(area,ue)
    return (alpha2_value * math.pow(t_list - tao_list, 0.8))/ (math.pow(t_list - tao_list, 0.8)+ 0.15 * th_value)


def get_k2(t_list,tao_list, area, ue): #handchecked
    k2 = np.zeros((len(tao_list), len(t_list)))
    for i in range(len(tao_list)):
        for j in range(len(t_list)):
            if t_list[j] == tao_list[i]: #loading = time
                k2[i, j] = 0

            elif t_list[j] < tao_list[i]: #time<loading eg Creep(time,loading) = Creep (1,2) = 0
                k2[i, j] = 0

            else:
                k2[i, j] = get_k2_value(area, ue, t_list[j], tao_list[i])
    return k2

def get_k3(tao_list):
    k3_array = np.zeros((len(tao_list), 1))
    for i in range(len(tao_list)):
        k3_array[i, 0] = 2.7/ (1 + math.log10(tao_list[i]))
    return k3_array

def get_k4(k4_key):
    environment_type = {0: 0.7, 1: 0.65, 2: 0.6, 3: 0.5} #Create a Dictionary for environment types
    return environment_type[k4_key]

def get_k5_value(f_c_key, area, ue):
    alpha2 = get_alpha2(area, ue)
    alpha3 = 0.7 / (k4 * alpha2)
    k5_value = (2 - alpha3) - 0.02 * (1 - alpha3) * f_c_key
    return k5_value

def get_k5(f_c_key, t_list, area, ue):
    k5_array = []
    if f_c_key <= 50:
        k5_output = np.ones((1, len(t_list)))
    elif 50 < f_c_input <= 100:
        for i in range(0, len(t_list)):
            k5_array.append(get_k5_value(f_c_key, area, ue))
        k5_output = np.array(k5_array)
    return k5_output

def get_phi_basic(f_c_key):
    creep_basic_array = np.array(
        [[20, 4.5], [25, 3.8], [32, 3], [40, 2.4], [50, 2], [65, 1.7], [80, 1.5], [100, 1.3]])
    for i in range(len(creep_basic_array)):
        if creep_basic_array[i][0] == f_c_key:
            p_b_output = creep_basic_array[i][1]
            break
        else:
            if creep_basic_array[i][0] < f_c_key:
                continue
            else:
                basic_creep_grad = (creep_basic_array[i][1] - creep_basic_array[i - 1][1]) \
                        / (creep_basic_array[i][0] - creep_basic_array[i - 1][0])
                p_b_output = creep_basic_array[i - 1][1] + (f_c_key - creep_basic_array[i - 1][0]) * basic_creep_grad
                break
    return p_b_output

def get_strain_cse(f_c_key, t_list):
    cse_array = []
    cse_star = (0.06 * f_c_key - 1) * 50 * math.pow(10, -6)
    for i in range(0, len(t_list)):
        cse_array.append(cse_star * (1 - math.exp(-0.1 * t_list[i])))
    return cse_array

def get_strain_shd_b_star():
    strain_input = get_single_input("Aggregate quality: good = 0, unknown = 1: ")
    if strain_input == 0:
        shd = 800 * math.pow(10, -6)
    elif strain_input == 1:
        shd = 1000 * math.pow(10, -6)
    else:
        print("Enter a valid number")
    return shd

def get_alpha1(area, ue):
    th = area / ue
    return 0.8 + 1.2 * math.exp(-0.005 * th)


def get_k1_value(area, ue, t, tao_d):
    th = area / ue
    a1 = get_alpha1(area, ue)
    return (a1 * math.pow(t - tao_d, 0.8))/ (math.pow(t - tao_d, 0.8) + 0.15 * th)

def get_k1(t_list, area, ue, tao_dry):
    k1_array = np.zeros((1, len(t_list)))
    for i in range(0, len(t_list)):
            k1_array[0, i] = get_k1_value(area, ue, t_list[i], tao_dry)
    return k1_array

def get_J_t_tao(t_list,tao_list, E_c_tao_array, phi_t_tao_array): #Note E_c_tao is dependant on time so refers to j #handchecked
    J_array = np.zeros((len(tao_list), len(t_list)))
    for i in range(0, len(tao_list)):
        for j in range(0, len(t_list)):
            if tao_list[i] > t_list[j] : # loading > time
                J_array[i, j] = 0
            else: # time >= loading
                J_array[i, j] = (1 / E_c_tao_array[i]) * (1 + phi_t_tao_array[i, j])
    return J_array

def get_E_c_j(t_list, tao_list, J_t_tao_array):
    E_c_j_array =np.zeros(len(t_list))
    for j in range(0, len(t_list)):
        for i in range(0, len(tao_list)):
            if t_list[j] == tao_list[i]: #loading = time
                E_c_j_array[j] = 1/ J_t_tao_array[i,j] #correct
                break #break I go to next j

            elif t_list[j] < tao_list[i]:  # if time<loading then use the previous tao value
                for a in range(0, len(t_list)):
                    if tao_list[i - 1] == t_list[a]:  # previous tao value (time>loading) is the same as the t value
                        E_c_j_array[j] = 1/ J_t_tao_array[i - 1, a]
                        break
                break
            elif i == len(tao_list)-1:
                for a in range(0, len(t_list)):
                    if tao_list[i] == t_list[a]:  # previous tao value (time>loading) is the same as the t value
                        E_c_j_array[j] = 1/ J_t_tao_array[i, a]
                        break
                break
    return E_c_j_array

def get_Fe_j_i(t_list, tao_list, J_t_tao_array):
    Fe_j_i = np.zeros((len(tao_list), len(t_list)))
    for j in range(0, len(t_list)):
        for tao_ip in range(0, len(tao_list)):
            if t_list[j] <= tao_list[tao_ip] or tao_ip == len(tao_list)-1:
                Fe_j_i[tao_ip,j] = 0
                break
            else:
                for i in range(0, len(tao_list)):
                    if t_list[j] < tao_list[i]:  # if time<loading then use the previous tao value
                     for a in range(0, len(t_list)):
                        if tao_list[i - 1] == t_list[a]:  # previous tao value (time>loading) is the same as the t value
                            Fe_j_i[tao_ip,j] = (J_t_tao_array[tao_ip+1,a]
                                                -J_t_tao_array[tao_ip,a])\
                                               /J_t_tao_array[i - 1, a]
                            break
                     break
                    elif i == len(tao_list)-1:
                        for a in range(0, len(t_list)):
                            if tao_list[i] == t_list[a]:  # previous tao value (time>loading) is the same as the t value
                                Fe_j_i[tao_ip,j] = (J_t_tao_array[tao_ip+1,a]
                                                    -J_t_tao_array[tao_ip,a])\
                                                   /J_t_tao_array[i, a]
                                break
                        break

    return Fe_j_i

def post_get_A_c(t_list, t_grout, area, d_duct, strands_arr, A_s_list, A_p_list):
    ac_array = []
    for i in range(0, len(t_list)):
        if t_list[i] >= t_grout: ## once grout is added
            a = area - sum(A_s_list) - sum(A_p_list)
        else:
            a = area - ((d_duct/2) ** 2 * math.pi) * sum(strands_arr) - sum(A_s_list)
        ac_array.append(a)
    output = np.array(ac_array)
    return output

def pre_get_A_c(t_list, area, A_s_list, A_p_list):
    ac_array = []
    for i in range(0, len(t_list)):
        a = area - sum(A_s_list) - sum(A_p_list)
        ac_array.append(a)
    output = np.array(ac_array)
    return output

def get_A_by_y(A_list, y_list, d_ref): #Used for S and P for sum of total layers
    B_mod = 0
    for i in range(0, len(A_list)):
        B_mod += A_list[i] * (y_list[i] - d_ref)
    return B_mod

def post_get_B_c(t_list, t_grout, area, d_cent, d_ref, d_duct, A_s_list, y_s_list,
                 A_p_list, y_p_list):
    bc_array = []
    for i in range(0, len(t_list)):
        if t_list[i] >= t_grout: ## once grout is added
            b = area * (d_cent - d_ref) \
                - get_A_by_y(A_s_list, y_s_list, d_ref) \
                - get_A_by_y(A_p_list, y_p_list, d_ref)
        else:   # made array of duct areas since get_A_by_y needs array, not single number
            b = area * (d_cent - d_ref) \
                - get_A_by_y(A_s_list, y_s_list, d_ref) \
                - get_A_by_y([((d_duct/2) ** 2 * math.pi)] * len(A_p_list), y_p_list, d_ref)
        bc_array.append(b)
    output = np.array(bc_array)
    return output

def pre_get_B_c(t_list, area, d_cent, d_ref, A_s_list, y_s_list,
                 A_p_list, y_p_list):
    bc_array = []
    for i in range(0, len(t_list)):
        b = area * (d_cent - d_ref) \
            - get_A_by_y(A_s_list, y_s_list, d_ref) \
            - get_A_by_y(A_p_list, y_p_list, d_ref)
        bc_array.append(b)
    output = np.array(bc_array)
    return output

def get_A_by_y2(A_list, y_list, d_ref): #Used for S and P for sum of total layers
    I_mod = 0
    for i in range(0, len(A_list)):
        I_mod += A_list[i] * ((y_list[i] - d_ref) ** 2)
    return I_mod

def post_get_I_c(I_g, t_list, t_grout, area, d_cent, d_ref, d_duct, A_s_list, y_s_list,
                 A_p_list, y_p_list):
    ic_array = []
    for i in range(0, len(t_list)):
        if t_list[i] >= t_grout: ## once grout is added
            ic = I_g + area * (d_cent - d_ref)**2 \
                - get_A_by_y2(A_s_list, y_s_list, d_ref) \
                - get_A_by_y2(A_p_list, y_p_list, d_ref)
        else:   # made array of duct areas since get_A_by_y needs array, not single number
            ic = I_g + area * (d_cent - d_ref)**2 \
                - get_A_by_y2(A_s_list, y_s_list, d_ref) \
                - get_A_by_y2([((d_duct/2) ** 2 * math.pi)] * len(A_p_list), y_p_list, d_ref)
        ic_array.append(ic)
    output = np.array(ic_array)
    return output

def pre_get_I_c(I_g, t_list, area, d_cent, d_ref, A_s_list, y_s_list,
                 A_p_list, y_p_list):
    ic_array = []
    for i in range(0, len(t_list)):
        ic = I_g + area * (d_cent - d_ref)**2 \
            - get_A_by_y2(A_s_list, y_s_list, d_ref) \
            - get_A_by_y2(A_p_list, y_p_list, d_ref)
        ic_array.append(ic)
    output = np.array(ic_array)
    return output

def get_A_c_slab(t_list, t_comp, area, A_s_list, A_p_list):
    ac_array = []
    for i in range(0, len(t_list)):
        if t_list[i] < t_comp:
            a = 0
        else:
            a = area - sum(A_s_list) - sum(A_p_list)
        ac_array.append(a)
    output = np.array(ac_array)
    return output

def get_B_c_slab(t_list, t_comp, area, d_cent, d_ref, A_s_list, y_s_list,
                 A_p_list, y_p_list):
    bc_array = []
    for i in range(0, len(t_list)):
        if t_list[i] < t_comp: # slab does not contribute since no comp action
            b = 0
        else:
            b = area * (d_cent - d_ref) \
                - get_A_by_y(A_s_list, y_s_list, d_ref) \
                - get_A_by_y(A_p_list, y_p_list, d_ref)
        bc_array.append(b)
    output = np.array(bc_array)
    return output

def get_I_c_slab(I_g, t_list, t_comp, area, d_cent, d_ref, A_s_list, y_s_list,
                 A_p_list, y_p_list):
    ic_array = []
    for i in range(0, len(t_list)):
        if t_list[i] < t_comp:
            ic = 0
        else:
            ic = I_g + area * (d_cent - d_ref)**2 \
                - get_A_by_y2(A_s_list, y_s_list, d_ref) \
                - get_A_by_y2(A_p_list, y_p_list, d_ref)
        ic_array.append(ic)
    output = np.array(ic_array)
    return output

def get_ABI_E(A_c_list, E_c_list):
    output = A_c_list * E_c_list
    return output

def get_AE(A_list, E_list):
    output = 0
    for i in range(0, len(A_list)):
        output += A_list[i] * E_list[i]
    return output

def get_R_A_j(t_list, t2_list, t_comp, A_c_list, E_c_list, A_c_list_2, E_c_list_2,
              A_s_list, E_s_list, A_s_list_2, E_s_list_2, A_p_list, E_p_list):
    array = np.zeros(len(t_list))
    for i in range(0, len(t_list)):
        if t_list[i] >= t_comp:
            array[i] = get_ABI_E(A_c_list[i], E_c_list[i]) \
                       + get_ABI_E(A_c_list_2[i - (len(t_list)-len(t2_list))], E_c_list_2[i - (len(t_list)-len(t2_list))]) \
                       + get_AE(A_s_list, E_s_list) + get_AE(A_s_list_2, E_s_list_2) \
                       + get_AE(A_p_list, E_p_list)
        else:
            array[i] = get_ABI_E(A_c_list[i], E_c_list[i]) \
                       + get_AE(A_s_list, E_s_list)
    return array

def get_YA_E(y_list, A_list, E_list):
    output = 0
    for i in range(0, len(A_list)):
        output += y_list[i] * A_list[i] * E_list[i]
    return output

def get_R_B_j(t_list, t2_list, t_comp, B_c_list, E_c_list, B_c_list_2, E_c_list_2,
              y_s_list, A_s_list, E_s_list, y_s_list_2, A_s_list_2, E_s_list_2, y_p_list, A_p_list, E_p_list):
    array = np.zeros(len(t_list))
    for i in range(0, len(t_list)):
        if t_list[i] >= t_comp:
            array[i] = get_ABI_E(B_c_list[i], E_c_list[i]) \
                       + get_ABI_E(B_c_list_2[i - (len(t_list)-len(t2_list))], E_c_list_2[i - (len(t_list)-len(t2_list))]) \
                       + get_YA_E(y_s_list, A_s_list, E_s_list) + get_YA_E(y_s_list_2, A_s_list_2, E_s_list_2) \
                       + get_YA_E(y_p_list, A_p_list, E_p_list)
        else:
            array[i] = get_ABI_E(B_c_list[i], E_c_list[i]) \
                       + get_YA_E(y_s_list, A_s_list, E_s_list)
    return array

def get_Y2A_E(y_list, A_list, E_list):
    output = 0
    for i in range(0, len(A_list)):
        output += y_list[i]**2 * A_list[i] * E_list[i]
    return output

def get_R_I_j(t_list, t2_list, t_comp, I_c_list, E_c_list, I_c_list_2, E_c_list_2,
              y_s_list, A_s_list, E_s_list, y_s_list_2, A_s_list_2, E_s_list_2, y_p_list, A_p_list, E_p_list):
    array = np.zeros(len(t_list))
    for i in range(0, len(t_list)):
        if t_list[i] >= t_comp:
            array[i] = get_ABI_E(I_c_list[i], E_c_list[i]) \
                       + get_ABI_E(I_c_list_2[i - (len(t_list)-len(t2_list))], E_c_list_2[i - (len(t_list)-len(t2_list))]) \
                       + get_Y2A_E(y_s_list, A_s_list, E_s_list) + get_Y2A_E(y_s_list_2, A_s_list_2, E_s_list_2) \
                       + get_Y2A_E(y_p_list, A_p_list, E_p_list)
        else:
            array[i] = get_ABI_E(I_c_list[i], E_c_list[i]) \
                       + get_Y2A_E(y_s_list, A_s_list, E_s_list)
    return array

def get_A_grout(d_duct, strands_arr, A_p_list):
    output = (d_duct/2) ** 2 * math.pi * sum(strands_arr) - sum(A_p_list)
    return output

def get_B_grout(d_duct, A_p_list, y_p_list):
    array = []
    for i in range(0, len(A_p_list)):
        output = ((d_duct/2) ** 2 * math.pi - A_p_list[i]) * y_p_list[i]
        array.append(output)
    return sum(array)

def get_I_grout(d_duct, A_p_list, y_p_list):
    array = []
    for i in range(0, len(A_p_list)):
        output = ((d_duct/2) ** 2 * math.pi - A_p_list[i]) * y_p_list[i]**2
        array.append(output)
    return sum(array)

def get_f_set_top(A_list, E_list, strain, B_list, curv): # for steel fset- top
    output = A_list * E_list * strain + B_list * E_list * curv
    return output

def get_f_set_bot(B_list, E_list, strain, I_list, curv): #for steep fset- bottom
    output = B_list * E_list * strain + I_list * E_list * curv
    return output

def get_f_set_conc(A_list, E_list, strain, B_list, curv, I_list): # calculate 2 x 1 fset for concrete elements
    # array = np.zeros((1,2))
    # array[0, 0] = get_f_set_top(A_list, E_list, strain, B_list, curv)
    # array[0, 1] = get_f_set_bot(B_list, E_list, strain, I_list, curv)
    array = []
    array.append(get_f_set_top(A_list, E_list, strain, B_list, curv))
    array.append(get_f_set_bot(B_list, E_list, strain, I_list, curv))

    return array

def get_f_set_grout(t_list, t_comp, A_list, E_list, strain, B_list, I_list): ## obtain single value, which will be modified to change over time (concrete)
    for i in range(0, len(t_list)):
        if t_list[i] > t_comp: # in case t_comp is not within the t array
            output = get_f_set_conc(A_list, E_list[i- 1], strain[0, i- 2], B_list, strain[1, i- 2], I_list)
            break

    return output

def get_f_set_slab(t_list, t2_list, t_comp, A_list, E_list, strain, B_list, I_list): # need to use strain[i] where i is based on t array not t2
    diff = len(t_list) - len(t2_list)
    for i in range(0, len(t_list)):
        if t_list[i] > t_comp: # in case t_comp is not within the t array
            output = get_f_set_conc(A_list[i - diff], E_list[i- diff- 1], strain[0, i- 2], B_list[i - diff], strain[1, i- 2], I_list[i - diff])
            break
    return output

def get_f_set(t_list, t_comp, A_list, E_list, strain, y_list): # 2x1 fset for each type of steel, constant over time
    for i in range(0, len(t_list)):
        if t_list[i] > t_comp: # in case t_comp is not within the t array
            top = 0
            bot = 0
            array = np.zeros((2, 1))
            for j in range(0, len(A_list)):
                b_calc = A_list[j] * y_list[j]
                i_calc = A_list[j] * y_list[j] **2
                top_calc = get_f_set_top(A_list[j], E_list[j], strain[0, i -2], b_calc, strain[1, i-2])
                bot_calc = get_f_set_bot(b_calc, E_list[j], strain[0, i- 2], i_calc, strain[1, i - 2])
                top += top_calc
                bot += bot_calc
            array[0, 0] = top
            array[1, 0] = bot
            break
    return array

def get_t_check(t_list, t_comp):
    array = []
    for i in range(0, len(t_list)):
        array.append(int(t_list[i] >= t_comp))
    return array

def get_E_c_conv(t_list, E_c_list, t_check_list):
    array = []
    for i in range(0, len(t_check_list)):
        if t_check_list[i] == 1:
            base = E_c_list[i]
            for j in range(0, len(t_list)):
                output = (E_c_list[j]/ base) * t_check_list[j]
                array.append(output)
            break
    return array

if __name__ == '__main__':
    tao = [7, 40, 40, 40, 60, 60, 30000] # hardcoded
    t = [7, 40, 40, 40, 60, 60, 30000]   # hardcoded

# steel reinf material properties
# moduli
E_s = [200000, 200000]        #hardcoded
E_p = [200000, 200000]          #hardcoded

# strain in steel at each t
f_p_steel = 1800    #hardcoded
sigma_jack = 1000   #hardcoded
rb_input = 1        #hardcoded
temp_input = 25     #hardcoded

r_b_steel = get_r_b_steel(rb_input)
k4_steel = np.array(get_k4_steel(t))
k5_steel = get_k5_steel(f_p_steel, sigma_jack, r_b_steel)
k6_steel = get_k6_steel(temp_input)

R_steel = k4_steel * k5_steel * k6_steel * r_b_steel
phi_p_steel = R_steel / (1 - R_steel)
phi_p_steel = np.array([0, 0.015, 0.015, 0.015, 0.02, 0.02, 0.03]) # hardcoded for example

# concrete element 1 - material properties
# E_c_tao calculation - p, fcmi, s
p_conc = 2400       #hardcoded
f_cmi_input = 50    #hardcoded
s_input = 0         #hardcoded

E_c = get_E_c(f_cmi_input, p_conc)
s = get_s(s_input)
E_c_tao = get_E_c_tao(t, s, E_c)
E_c_tao = np.array([25 * math.pow(10, 3), 32 * math.pow(10, 3), 32 * math.pow(10, 3),
                    32 * math.pow(10, 3), 34 * math.pow(10, 3), 34 * math.pow(10, 3),
                    38 * math.pow(10, 3)]) # hardcoded

# phi_t_tao calculation - k2, k3, k4, k5, phi_basic
k4_input = 0    #hardcoded
f_c_input = 60  #hardcoded
area_1 = 317000 #hardcoded
ue_1 = 2200     #hardcoded

k2 = get_k2(t, tao, area_1, ue_1)
k3 = get_k3(tao)
k4 = get_k4(k4_input)
k5 = get_k5(f_c_input, t, area_1, ue_1)
phi_basic = get_phi_basic(f_c_input)

phi_t_tao = k2 * k3 * k4 * k5 * phi_basic
phi_t_tao = np.array([[0, 0.8, 0.8, 0.8, 1, 1, 2.5],
                      [0, 0, 0, 0, 0.8, 0.8, 1.8],
                      [0, 0, 0, 0, 0.8, 0.8, 1.8],
                      [0, 0, 0, 0, 0.8, 0.8, 1.8],
                      [0, 0, 0, 0, 0, 0, 1.6],
                      [0, 0, 0, 0, 0, 0, 1.6],
                      [0, 0, 0, 0, 0, 0, 0]]) # hardcoded

# J_t_tao, E_c_j, Feji
J_t_tao = get_J_t_tao(t, tao, E_c_tao, phi_t_tao)
E_c_j = get_E_c_j(t, tao, J_t_tao)
F_e_j_i = get_Fe_j_i(t, tao, J_t_tao)

# shrinkage strain over time
strain_cse = get_strain_cse(f_c_input, t)
# strain_shd_b_star = get_strain_shd_b_star() # bring this back
strain_shd_b_star = 0.0008 # hard coded
strain_shd_b = (1 - 0.008 * f_c_input) * strain_shd_b_star

tao_dry_1 = 7 # hard coded - should be 1 x n where n is number of components
k1 = get_k1(tao, area_1, ue_1, tao_dry_1)
strain_shd = k1 * k4 * strain_shd_b
strain_sh_j = strain_cse + strain_shd
strain_sh_j = np.array([0, -100 * math.pow(10, -6), -100 * math.pow(10, -6),
                        -100 * math.pow(10, -6), -150 * math.pow(10, -6),
                        -150 * math.pow(10, -6), -400 * math.pow(10, -6)]) # hardcoded

# prestress information
P_p_init = 1000000 #hardcoded, force per strand
n_strands = np.array([1, 1], dtype=np.float64) # hardcoded, strands per row
y_p = [1030, 1160] #hardcoded, prestress distance from very top

f_p_init_N = get_f_p_init_N(t, n_strands, P_p_init)
f_p_init_M = get_f_p_init_M(t, n_strands, P_p_init, y_p)
f_p_init = np.array([f_p_init_N, f_p_init_M])

f_p_rel_j = get_f_p_rel(f_p_init_N, f_p_init_M, phi_p_steel)

t_grout = 41    ## hardcoded, is when grout is added
t_comp = 41    ## hardcoded, is when the grout and deck is set so comp action is achieved

# Ac, Bc, Ic of section variables required
A_s = [900, 1800]       # hardcoded
A_p = [800, 800]        # hardcoded
y_s = [210, 1240]           # hardcoded: y_p is above under prestress properties
diam_duct = 60              # hardcoded
d_c = 752                   # hardcoded
d_ref = 0                   # hardcoded
i_gross = 49900 * 10 ** 6   # hardcoded

pre_A_c = pre_get_A_c(t, area_1, A_s, A_p)
pre_B_c = pre_get_B_c(t, area_1, d_c, d_ref, A_s, y_s, A_p, y_p)
pre_I_c = pre_get_I_c(i_gross, t, area_1, d_c, d_ref, A_s, y_s, A_p, y_p)

post_A_c = post_get_A_c(t, t_grout, area_1, diam_duct, n_strands, A_s, A_p)
post_B_c = post_get_B_c(t, t_grout, area_1, d_c, d_ref, diam_duct, A_s, y_s, A_p, y_p)
post_I_c = post_get_I_c(i_gross, t, t_grout, area_1, d_c, d_ref, diam_duct, A_s, y_s, A_p, y_p)

post_A_c = np.array([308645.13, 308645.13, 308645.13,
                     312700, 312700, 312700, 312700]) # hardcoded
post_B_c = np.array([2.2977e+08, 2.2977e+08, 2.2977e+08,
                     2.3421e+08, 2.3421e+08, 2.3421e+08, 2.3421e+08 ]) # hardcoded
post_I_c = np.array([2.1955e+11, 2.1955e+11, 2.1955e+11,
                     2.2443e+11, 2.2443e+11, 2.2443e+11, 2.2443e+11]) # hardcoded

# concrete element 2 - material properties

t_comp_2 = 40
t_2 = [40, 60, 60, 30000]       # hardcoded - t array but only show post slab pour times
tao_2 = [40, 60, 60, 30000]
t_grout_2 = 40
area_2 = 360000
ue_2 = 3500     #hardcoded
diam_duct_2 = 0
n_strands_2 = [0]
A_s_2 = [2700]
E_s_2 = [200000]
y_s_2 = [75]
A_p_2 = [0]
E_p_2 = [0]
y_p_2 = [0]
d_c_2 = 75
i_gross_2 = 675 * 10 ** 6
k4_input_2 = 0    #hardcoded
f_c_input_2 = 60  #hardcoded

# E_c_tao_2 calculation - p, fcmi, s
p_conc_2 = 2400       #hardcoded
f_cmi_input_2 = 50    #hardcoded
s_input_2 = 0         #hardcoded

E_c_2 = get_E_c(f_cmi_input_2, p_conc_2)
s_2 = get_s(s_input_2)
t_slab = 40

E_c_tao_2_t = get_E_c_tao_2_t(t_2, t_slab) ## array with days since slab pour
E_c_tao_2 = get_E_c_tao(E_c_tao_2_t, s_2, E_c_2)
E_c_tao_2 = np.array([18 * math.pow(10, 3), 25 * math.pow(10, 3),
                      25 * math.pow(10, 3), 30 * math.pow(10, 3)]) # hardcoded

k2_2 = get_k2(t_2, tao_2, area_2, ue_2)
k3_2 = get_k3(tao_2)
k4_2 = get_k4(k4_input_2)
k5_2 = get_k5(f_c_input_2, t_2, area_2, ue_2)
phi_basic_2 = get_phi_basic(f_c_input_2)
phi_t_tao_2 = k2_2 * k3_2 * k4_2 * k5_2 * phi_basic_2
phi_t_tao_2 = np.array([[0, 2, 2, 3.5],
                      [0, 0, 0, 2.8],
                      [0, 0, 0, 2.8],
                      [0, 0, 0, 0]]) # hardcoded

# shrinkage strain over time
strain_cse_2 = get_strain_cse(f_c_input_2, t_2)

# strain_shd_b_star = get_strain_shd_b_star() # bring this back
strain_shd_b_star_2 = 0.0008 # hard coded
strain_shd_b_2 = (1 - 0.008 * f_c_input_2) * strain_shd_b_star_2

tao_dry_2 = 7 # hard coded - should be 1 x n where n is number of components
k1_2 = get_k1(tao_2, area_2, ue_2, tao_dry_2)
strain_shd_2 = k1_2 * k4_2 * strain_shd_b
strain_sh_j_2 = strain_cse_2 + strain_shd_2

strain_sh_j_2 = np.array([0,-200 * math.pow(10,-6),
                          -200 * math.pow(10,-6), -600 * math.pow(10,-6),])

# J_t_tao, E_c_j, Feji
J_t_tao_2 = get_J_t_tao(t_2, tao_2, E_c_tao_2, phi_t_tao_2)
E_c_j_2 = get_E_c_j(t_2, tao_2, J_t_tao_2)
F_e_j_i_2 = get_Fe_j_i(t_2, tao_2, J_t_tao_2)

# slab (2) Ac, Bc and Ic is not affected by post/pre (no grout in slab)
# will be 0 if composite action is not available at t
A_c_2 = get_A_c_slab(t_2, t_comp_2, area_2, A_s_2, A_p_2)
B_c_2 = get_B_c_slab(t_2, t_comp_2, area_2, d_c_2, d_ref, A_s_2, y_s_2, A_p_2, y_p_2)
I_c_2 = get_I_c_slab(i_gross_2, t_2, t_comp_2, area_2, d_c_2, d_ref, A_s_2, y_s_2, A_p_2, y_p_2)

A_c_2 = np.array([357300, 357300, 357300, 357300])
B_c_2 = np.array([26797500, 26797500, 26797500, 26797500])
I_c_2 = np.array([2684812500, 2684812500, 2684812500, 2684812500])

# cross sectional rigidities
# post because post_A_c, if want pre, use pre_A_c
post_R_A_j = get_R_A_j(t, t_2, t_comp_2, post_A_c, E_c_tao, A_c_2, E_c_tao_2, A_s, E_s, A_s_2, E_s_2, A_p, E_p)
post_R_A_j = np.array([8.2561e+09, 1.0417e+10, 1.0417e+10, 1.7838e+10, 2.0964e+10, 2.0964e+10, 2.4002e+10]) # hardcoded
post_R_B_j = get_R_B_j(t, t_2, t_comp_2, post_B_c, E_c_tao, B_c_2, E_c_tao_2, y_s, A_s, E_s, y_s_2, A_s_2, E_s_2, y_p, A_p, E_p)
post_R_B_j = np.array([6.2284e+12, 7.8368e+12, 7.8368e+12, 8.8522e+12, 9.5082e+12, 9.5082e+12, 1.0579e+13]) # hardcoded
post_R_I_j = get_R_I_j(t, t_2, t_comp_2, post_I_c, E_c_tao, I_c_2, E_c_tao_2, y_s, A_s, E_s, y_s_2, A_s_2, E_s_2, y_p, A_p, E_p)
post_R_I_j = np.array([6.0502e+15, 7.58710e+15, 7.5871e+15, 8.1796e+15, 8.6473e+15, 8.6473e+15, 9.5584e+15])

r_e_j = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [550 * math.pow(10,6), 550 * math.pow(10,6),
                   1170 * math.pow(10,6), 1170 * math.pow(10,6), 1170 * math.pow(10,6),
                   1570 * math.pow(10,6), 1570 * math.pow(10,6)]]) #hardcoded

# create Fj
def get_F_j(t_list, Ra, Rb, Ri):
    fj_a = []
    fj_b = []
    fj_i = []
    for i in range(0, len(t_list)):
        p = 1/(Ra[i] * Ri[i] - Rb[i] ** 2)
        fj_a.append(p * Ra[i])
        fj_b.append(p * Rb[i])
        fj_i.append(p * Ri[i])
    array = np.array([[fj_i[0], -fj_b[0]], [-fj_b[0], fj_a[0]]])

    for i in range(1, len(t_list)):  # appending 2 x 2 arrays for original 2x2 Fj array
        a = np.array([[fj_i[i], -fj_b[i]], [-fj_b[i], fj_a[i]]])
        array = np.concatenate((array, a), axis=1)
    return array

post_F_j = get_F_j(t, post_R_A_j, post_R_B_j, post_R_I_j)

def get_ABE_strain(AB_list, E_list, strain_list):
    output = AB_list * E_list * strain_list
    return output

# def get_f_sh_j(t_list, t2_list, A_c_list, E_c_list, B_c_list, strain_list,
#                A_c_list_2, E_c_list_2, B_c_list_2, strain_list_2):
#     array = np.zeros((2, len(t_list)))
#     for i in range(0,len(t_list)):
#         if t_list[i] > t_comp:
#             array[0, i] = get_ABE_strain(A_c_list[i], E_c_list[i], strain_list[i]) \
#                           + get_ABE_strain(A_c_list_2[i - (len(t_list)-len(t2_list))],
#                                            E_c_list_2[i - (len(t_list)-len(t2_list))],
#                                            strain_list_2[i - (len(t_list)-len(t2_list))])
#             array[1, i] = get_ABE_strain(B_c_list[i], E_c_list[i], strain_list[i]) \
#                           + get_ABE_strain(B_c_list_2[i - (len(t_list)-len(t2_list))],
#                                            E_c_list_2[i - (len(t_list)-len(t2_list))],
#                                            strain_list_2[i - (len(t_list)-len(t2_list))])
#         else:
#             array[0, i] = get_ABE_strain(A_c_list[i], E_c_list[i], strain_list[i])
#             array[1, i] = get_ABE_strain(B_c_list[i], E_c_list[i], strain_list[i])
#     return array

def get_f_sh_j(t_list, A_c_list, E_c_list, B_c_list, strain_list):
    array = np.zeros((2, len(t_list)))
    for i in range(0, len(t_list)):
        array[0, i] = get_ABE_strain(A_c_list[i], E_c_list[i], strain_list[i])
        array[1, i] = get_ABE_strain(B_c_list[i], E_c_list[i], strain_list[i])
    return array

def get_f_sh_j_slab(t_list, t2_list, A_c_list, E_c_list, B_c_list, strain_list):
    array = np.zeros((2, len(t2_list)))
    for i in range(0, len(t2_list)):
        array[0, i] = get_ABE_strain(A_c_list[i], E_c_list[i], strain_list[i])
        array[1, i] = get_ABE_strain(B_c_list[i], E_c_list[i], strain_list[i])
    start = np.zeros((2, len(t_list) - len(t2_list)))
    array = np.concatenate((start, array), axis = 1)
    return array

post_f_sh_j = get_f_sh_j(t, post_A_c, E_c_tao, post_B_c, strain_sh_j)
post_f_sh_j_slab = get_f_sh_j_slab(t, t_2, A_c_2, E_c_tao_2, B_c_2, strain_sh_j_2)

# for pre, use pre_A_c etc
# post_f_sh_j = get_f_sh_j(t, t_2, post_A_c, E_c_tao, post_B_c, strain_sh_j,
#                A_c_2, E_c_tao_2, B_c_2, strain_sh_j_2)
# post_f_sh_j = np.array([[0, -9.8766e+05, -9.8766e+05, -9.8766e+05, -3.3813e+06, -3.3813e+06, -1.1184e+07],
#                        [0, -7.3526e+08, -7.3526e+08, -7.3526e+08, -1.3285e+09, -1.3285e+09, -4.04235e+09]])




def get_D_c_j(t_list, A_c_list, B_c_list, I_c_list, E_c_list):
    array = E_c_list[0] * np.array([[A_c_list[0], B_c_list[0]], [B_c_list[0], I_c_list[0]]])
    for i in range(1, len(t_list)):
        a = E_c_list[i] * np.array([[A_c_list[i], B_c_list[i]], [B_c_list[i], I_c_list[i]]])
        array = np.concatenate((array, a), axis=1)
    return array
D_c_j = get_D_c_j(t, post_A_c, post_B_c, post_I_c, E_c_tao)
D_c_j_2 = get_D_c_j(t_2, A_c_2, B_c_2, I_c_2, E_c_tao_2)


strain_j = np.zeros((2, len(t)))
f_cr_j = np.zeros((2, len(t)))
r_c_j_1 = np.zeros((2, len(t)))
f_set_j = np.zeros((2,len(t)))

f_cr_j_2 = np.zeros((2, len(t_2)))
r_c_j_2 = np.zeros((2, len(t_2)))

post_tension = 1 #hardcoded
m = 0
n = 2
t = [7, 39, 39, 40, 60, 60, 30000]
f_set_c_slab = np.zeros((2, len(t)))
f_set_c_grout = np.zeros((2, len(t)))
f_set_j = np.zeros((2, len(t)))

t_check_1 = get_t_check(t, t_comp_2)
E_c_conv_1 = get_E_c_conv(t, E_c_tao, t_check_1)
t_check_2 = get_t_check(t_2, t_comp_2)
E_c_conv_2 = get_E_c_conv(t_2, E_c_tao_2, t_check_2)

for i in range(0, len(t)):
    output = 0

    for j in range(0, len(tao)):
        d = F_e_j_i[j, i] * r_c_j_1[:, j]
        output = output + d
    f_cr_j[:, i] = output

    if t[i] < t_comp_2: # reassigned t array to make it easier to work with, remove it later
        # f_set_slab = 0

        a = (r_e_j[:, i] - f_cr_j[:, i] + post_f_sh_j[:, i] - f_p_init[:, i] + f_p_rel_j[:, i]).reshape(-1, 1)
        strain_j[:, i] = np.dot(post_F_j[:, m:n], a).reshape( 1, -1)
        r_c_j_1[:, i] = np.dot(D_c_j[:, m:n], strain_j[:,i]) + f_cr_j[:,i] - post_f_sh_j[:, i]

        m = m + 2
        n = n + 2
    else: # whole array is replaced every iteration instead

        f_set_p = get_f_set(t, t_comp_2, A_p, E_p, strain_j, y_p)  # prestress in element 1
        f_set_s_2 = get_f_set(t, t_comp_2, A_s_2, E_s_2, strain_j, y_s_2)  # steel in element 2
        f_set_c_slab[:, i] = get_f_set_slab(t, t_2, t_comp_2, A_c_2, E_c_tao_2, strain_j, B_c_2,
                                            I_c_2) # concrete in slab (element 2)
        f_set_c_slab[:, i] = f_set_c_slab[:, i] * E_c_conv_2[i - (len(E_c_tao) - len(E_c_tao_2))]

        if post_tension == 1:
            area_grout = get_A_grout(diam_duct, n_strands, A_p)
            B_grout = get_B_grout(diam_duct, A_p, y_p)
            I_grout = get_I_grout(diam_duct, A_p, y_p)
            f_set_c_grout[:, i] = get_f_set_grout(t, t_comp_2, area_grout, E_c_tao, strain_j, B_grout,
                                            I_grout)  # grout in section (element 1)
            f_set_c_grout[:, i] = f_set_c_grout[:, i] * E_c_conv_1[i]

        else:
            area_grout = 0
            B_grout = 0
            I_grout = 0

        f_set_j[:, i] = (f_set_p + f_set_s_2).reshape(1, -1) + f_set_c_slab[:, i] + f_set_c_grout[:, i]

        a = (r_e_j[:, i] - f_cr_j[:, i] + post_f_sh_j[:, i] + post_f_sh_j_slab[:, i] - f_p_init[:, i] + f_p_rel_j[:, i] + f_set_j[:, i]).reshape(-1, 1)
        strain_j[:, i] = np.dot(post_F_j[:, m:n], a).reshape(1, -1)

        r_c_j_1[:, i] = np.dot(D_c_j[:, m:n], strain_j[:, i]) + f_cr_j[:, i] - post_f_sh_j[:, i]  - f_set_c_grout[:, i]

        r_c_j_2[:, i - (len(t)-len(t_2))] = np.dot(D_c_j[:, m - 2*i:n - 2*i], strain_j[:, i]) + f_cr_j[:, i] - post_f_sh_j_slab[:, i] - f_set_c_slab[:, i]
        # fix rcj2
        m = m + 2
        n = n + 2

print(r_c_j_1)
print(r_c_j_2)





# aoa = [0,0,0,0,0,0,0]
# print(len(aoa))
# bob = np.zeros((2,7))
# print(bob)
# for i in range(0, len(aoa)):
#     bob[:, i] = [50, 60]
# print(bob)


