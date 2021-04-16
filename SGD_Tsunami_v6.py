#!/usr/bin/env python
# coding: utf-8

# # Implementation of the stochastic gradiant descent algorithm
# # https://en.wikipedia.org/wiki/Stochastic_gradient_descent
# # And ADAM Gradient Descent
# # https://arxiv.org/pdf/1412.6980.pdf
# # Coupled to Scale Tsunami 

# In[1]:


import os
import scale_file_handler
import math
from scipy.linalg import null_space
import numpy as np
import random


# In[2]:


### This function turns a set a of beta values into a runable scale input
def build_scale_input_from_beta(scale_handler,
                                material_betas,
                                material_1,
                                material_2,
                                template_file_string,
                                flag,
                                flag_replacement_string='replace',
                                temperature=300,
                                material_count_offset=1,
                                file_name_flag='default_',
                                replacement_dict_addition = ''):

    material_list = []
    for beta in material_betas:
        material_list.append(scale_handler.combine_material_dicts(material_1, material_2, beta))

    material_string_list = []
    for count, material in enumerate(material_list):
        material_string_list.append(
            scale_handler.build_scale_material_string(material, count + material_count_offset, temperature))

    ### Making list of keys
    flag_list = []
    for x in range(len(material_string_list)):
        flag_list.append(flag.replace(flag_replacement_string, str(x)))

    material_dict = scale_handler.make_data_dict(flag_list, material_string_list)
    
    for flag in replacement_dict_addition:
        material_dict[flag] = replacement_dict_addition[flag]

    scale_handler.create_scale_input_given_target_dict(template_file_string, file_name_flag, material_dict)

### This function takes a list of materials in each material type and sums
### the sensitivites for each. 
### Inputs:
### materials_list - list of dictionaries in with the form {"isotope":nuclear density,...}
### sensitivities - nested dictionaries
def combine_sensitivities_by_list(materials_list, sensitivities):
    #print(materials_list)
    #print(sensitivities)
    material_sens_lists = []

    ### Sum all sensitivities for each material dictionary in the list of materials
    for material_dict in materials_list:
        ### Sum all poison and fuel/mod sensitivities
        sensitivity_sum_list = []
        for material_loc in sensitivities:

            if material_loc == '0':
                #print("SKIPPING TOTAL SENSITIVITY")
                continue

            sum_ = 0.0

            for isotope in material_dict:
                try:
                    sum_ += float(sensitivities[material_loc][isotope]['sensitivity'])
                except:
                    print("WARNING: Missing sensitivities for: " + isotope +                           " If this is not in the first step there's a problem. Otherwise, the default sdf may not have all of the isotopes required. ")
            sensitivity_sum_list.append(sum_)

        material_sens_lists.append(sensitivity_sum_list)

    return material_sens_lists

### This function takes the d%keff/d%material_change and turns them into d%keff/dbeta
### Inputs:
### tsunami_betas: List of beta values from 0-1
### material_1 and 2_sensitivities: the total tsunami sensitivities from calculation
### beta_diff is the amount added and subtracted to beta values
def calculate_sensitivities_2_materials_general(tsunami_betas,
                                                material_1_sensitivities,
                                                material_2_sensitivities,
                                                beta_diff = 0.01):
    sensitivities = []
    for mat_count, material_1_sensitivity in enumerate(material_1_sensitivities):
        material_2_sensitivity = material_2_sensitivities[mat_count]
        beta_ = tsunami_betas[mat_count]

        ### Calculating percent change in poison
        ###     Calculating % change in each material
        x_1_material_1_beta_change_percent = (beta_ + beta_diff) / beta_ - 1
        
        x_1_material_2_beta_change_percent = (1 - beta_ - beta_diff) / (1 - beta_) - 1

        x_2_material_1_beta_change_percent = (beta_ - beta_diff ) / beta_ - 1
        x_2_material_2_beta_change_percent = (1 - beta_ + beta_diff) / (1 - beta_) - 1
        

        ###     Multiplying the percent change in beta by the sensitivity 
        ###     per % change in beta.
        y_1 = x_1_material_1_beta_change_percent * material_1_sensitivity +               x_1_material_2_beta_change_percent * material_2_sensitivity

        y_2 = x_2_material_1_beta_change_percent * material_1_sensitivity +               x_2_material_2_beta_change_percent * material_2_sensitivity

        ###    
        x_1 = beta_ + beta_diff
        x_2 = beta_ - beta_diff
        
        ### Adding calculating d% sensitivity/dbeta 
        sensitivities.append((y_2 - y_1) / (x_2 - x_1))
    return sensitivities




### This function takes the material betas that describe the geometry, builds the tsunami job and runs it.
### Then it pulls out the keff and senstivities and converts them into usable form
### Inputs:
### material_betas - list of beta values (between 0-1)
### materials - list of material dictionaries for each material
### tsunami_job_flag - string to start tsunami jobs with
### debug_fake_tsunami_run - skips running tsunami 
def evaluate_with_Tsunami(material_betas,
                          materials,
                          tsunami_job_flag = "tsunami_job",
                          build_input = True,
                          submit_tsunami_job = True,
                          pull_keff = True,
                          pull_sensitivities = True):
    
    sfh = scale_file_handler.scale_file_handler()
    default_material_list = sfh.build_material_dictionaries(materials, multiplier = 1.0)
    
    ### Building scale input
    if build_input:
        random_number = random.randint(1152921504606846976,18446744073709551615)
        hex_number = str(hex(random_number))
        hex_number = hex_number [2:]
        rep_dict_addition = {'%%%random_number%%%':hex_number}

        build_scale_input_from_beta(sfh,
                                     material_betas=material_betas,
                                     material_1=default_material_list[0],
                                     material_2=default_material_list[1],
                                     flag="%material_replace%",
                                     flag_replacement_string='replace',
                                     template_file_string="tsunami_template_file_11x11.inp",
                                     file_name_flag=tsunami_job_flag,
                                     replacement_dict_addition = rep_dict_addition)
        sfh.build_scale_submission_script(tsunami_job_flag, solve_type = 'tsunami')
        
    else:
        print("Skipping building scale input file.")
    

    if submit_tsunami_job:
        assert build_input == True, "You didn't build a new input, but you're submitting to the cluster. Quite irregular."
        sfh.submit_jobs_to_necluster(tsunami_job_flag)
        sfh.wait_on_submitted_job(tsunami_job_flag)
    else:
        print("Not submitting the tsunami job.")
    
    ### Pulling out keff from Tsunami job
    if pull_keff:
        print("    Pulling keff")
        ### Checking if tsunami_jog_flag has ".out" at the end, if not, add it.
        if tsunami_job_flag.endswith('.out') == False:
            keff_filename = tsunami_job_flag + ".out"
        else:
            keff_filename = tsunami_job_flag
            
        keff, uncert = sfh.get_keff_and_uncertainty(keff_filename)
    else:
        print("Faking keff")
        keff = 0.0
    
    ### Pulling out sensitivities and turning them into dk/k/dB 
    if pull_sensitivities:
        print("    Pulling sensitivities")
        
        ### Checking if tsunami_jog_flag has ".sdf" at the end, if not, add it.
        if tsunami_job_flag.endswith('.sdf') == False:
            sdf_filename = tsunami_job_flag + ".sdf"
        else:
            sdf_filename = tsunami_job_flag
        
        material_sensitivites = combine_sensitivities_by_list(default_material_list,
                                                              sfh.parse_sdf_file_into_dict(sdf_filename))

        beta_sensitivities    = calculate_sensitivities_2_materials_general(material_betas,
                                                material_sensitivites[0],
                                                material_sensitivites[1])
        print("Returning keff and sensitivities sensitivities ")
    else:
        print("Faking sensitivities")
        material_sensitivites = [1.0, 1.0]
        
    return keff, beta_sensitivities, material_sensitivites[0], material_sensitivites[1]

def pull_sensitivities():
    default_material_list = sfh.build_material_dictionaries(materials, multiplier = 1.0)
    material_sensitivites = combine_sensitivities_by_list(default_material_list,
                                                          sfh.parse_sdf_file_into_dict(tsunami_job_flag + ".sdf"))

    return calculate_sensitivities_2_materials_general(material_betas,
                                            material_sensitivites[0],
                                            material_sensitivites[1])
# In[5]:


### inputs:
### x_dim, y_dim - X and Y size of beta matrix
### build_type - "fixed" for applying "fixed_value" to each location, "random" for uniformly distributed random values 
def build_initial_betas(x_dim, y_dim, build_type, rand_min = 0.0, rand_max = 1.0, fixed_value = 0.5):
    material_betas = []
    for x in range(x_dim):
        for y in range(y_dim):
            if build_type == 'random':
                material_betas.append(random.uniform(rand_min, rand_max))
            if build_type == 'fixed':
                material_betas.append(fixed_value)
    return material_betas


# In[6]:


### This function takes the gradient descent step. checks if the values stay between 0.99 and 0.01
### inputs:
### variables
### negative sensitivities
### step_size
def calculate_new_variables(variables, negative_sensitivities, step_size):
    new_variables = [float(variable_ + step_size * deriv_) for variable_, deriv_ in zip(variables, negative_sensitivities)]
    
    ### Checking if variables meet variable requirement
    new_new_variables = []
    for variable in new_variables:
        if variable > 1.0:
            variable = 0.99
        if variable < 0.0:
            variable = 0.01
        new_new_variables.append(variable)
        
    return new_new_variables

### Implementation of vanilla gradient descent
def gradient_descent_scale(initial_betas,
                           stopping_value = 0.001,
                           number_of_steps = 10,
                           step_size_type = 'fixed#0.1', 
                           debug_fake_tsunami_run = False,
                           debug_print_betas = False,
                          write_output = False,
                          write_output_string = "output.csv",
                          null_space_adj = False,
                          materials = ["void", "fuel"]):
    steps = 1
    stopping_criteria = False
    variables = initial_betas
    
    if write_output:
        with open(write_output_string, 'w') as output_file:
            output_file.write("step, keff, step_size, old_betas, new_betas, negative_sense, beta_sensitivities\n")
    
    if null_space_adj:
        ss = null_space(np.ones((1,121)))
        proj = np.matmul(ss, np.transpose(ss))

    while steps < number_of_steps+1 and stopping_criteria == False:
        print("Step #:", steps)
        ### 
        tsunami_job_flag = 'tsunami_job_' + str(steps)
        
        if 'fixed' in step_size_type:
            step_size_ = step_size_type.split("#")
            step_size = float(step_size_[1])
        if 'sqrt_n' in step_size_type:
            step_size = 1/math.sqrt(steps)
            if 'mult' in step_size_type:
                mult_ = step_size_type.split("#")
                step_size *= float(mult_[1])
        
        ### Evaluate with TSUNAMI
        keff, beta_sensitivities, material_1_sense, material_2_sense = evaluate_with_Tsunami(variables,
                                                             tsunami_job_flag = tsunami_job_flag,                    
                                                             debug_fake_tsunami_run = debug_fake_tsunami_run,
                                                             materials = materials)
        print(beta_sensitivities[0])
        if null_space_adj:
            #print("pre beta sensitivities\n",np.array(beta_sensitivities))
            beta_sensitivities = np.matmul(np.array(beta_sensitivities), proj)
            #print("beta sensitivities\n",np.array(beta_sensitivities))
            #print("proj\n",proj)
        
        ### Mulitplying derivatives by keff
        negative_sensitivities = [float(deriv  * float(keff)) for deriv in beta_sensitivities]
        if debug_print_betas:
            print(negative_sensitivities)
        print("Calculating step, step size:", step_size)
        #new_variables = [float(variable_ - step_size * deriv_) for variable_, deriv_ in zip(variables, negative_sensitivities)]
        new_variables = calculate_new_variables(variables, negative_sensitivities, step_size)
        if debug_print_betas:
            print(new_variables)
        #new_variable = variables - step_size * function_derivative(variables)
        #stopping_criteria = stopping_criteria_function(variables, new_variables, stopping_value)
        
        
        
        if write_output:
            with open(write_output_string, 'a') as output_file:
                write_string = str(steps) + "," + str(keff) + "," + str(step_size)
                
                for _ in variables:
                    write_string += "," + str(_)
                
                for _ in new_variables:
                    write_string += "," + str(_)
                
                for _ in negative_sensitivities:
                    write_string += "," + str(_)
                    
                for _ in beta_sensitivities:
                    write_string += "," + str(_)
                    
                for _ in material_1_sense:
                    write_string += "," + str(_)
                    
                for _ in material_2_sense:
                    write_string += "," + str(_)
                
                output_file.write(write_string + "\n")
        steps += 1        
        variables = new_variables


# In[7]:


#gradient_descent_scale(material_betas,
#                       debug_fake_tsunami_run = True,
#                       debug_print_betas = True,
#                       step_size_type = 'sqrt_n_mult#1',
#                       number_of_steps = 10,
#                       write_output = True,
#                       null_space_adj = True,
#                      materials = ["void", "fuel/moderator:25/75"])


# In[8]:


### Implementation of ADAM gradient descent
### https://arxiv.org/pdf/1412.6980.pdf

### Function which takes the list of beta values, checks to see if the values are above or below limits and sets them to
### those limits
def check_beta_values(betas, min_val = 0.01, max_val = 0.99):
    new_betas = []
    for val in betas:
        if val > max_val:
            val = max_val
        if val < min_val:
            val = min_val
        new_betas.append(val)
    return new_betas 

def fix_mass(betas, target_mass, min_val = 0.01, max_val = 0.99, sticky_mass = True):
    current_mass = sum(betas)
    adjustment_factor = target_mass / current_mass 
    new_betas = []
    for _ in betas:
        if sticky_mass:
            if (_ == min_val):
                new_betas.append(_)
                continue
            elif (_ == max_val):
                new_betas.append(_)
                continue
        
        new_betas.append(_*adjustment_factor)
    return new_betas

### Function which takes betas, target_mass, whether to use "sticky values" (make highest and lowest values unchanged)
def fixed_mass_adjustment(betas, target_mass, sticky_mass = True, debug=False, mass_round_dig = 5):
    if debug:
        print("Curent mass: {}, target: {}".format(sum(betas), target_mass))
    material_betas = check_beta_values(betas)
    while round(sum(material_betas), 5) != 60.5:
        material_betas = fix_mass(betas = material_betas, target_mass = 60.5, sticky_mass = sticky_mass)
        material_betas = check_beta_values(material_betas)
        if debug:
            print(sum(material_betas))
    return material_betas   

### Implementation of ADAM gradient descent
### inputs:
### material betas - list of values from 0-1.0 describing material
### debug_fake_tsunami_run - Boolean, if True running Tsunami is skipped
### debug_print_betas- Boolean, if true beta values are printed each step
### number_of_steps - Int, total number of steps to take with algo
### alpha_value - Float, Step size, set to default value
### beta_1 - Float, Decay rate for first moment, set to default value
### beta_2 - Float, Decay rate for second moment, set to default value
### epsilon - Float, Small value used to avoid division by zero, set to default value
### write_output - Boolean, True to write out output
### null_space_adj - Boolean, if True multiply penultimate values by null vector so that their changes sum to 0
### materials - List of materials void, (TCR) fuel and moderator. If you want a mixed material the form is:
###    "material 1 string/material 2 string:fraction material 1/fraction material 2"
def adam_gradient_descent_scale(initial_betas,
                           debug_print_all = False,
                           submit_tsunami_job = True,
                           stopping_value = 0.001,
                           number_of_steps = 10,
                           alpha_value = 0.1, 
                           beta_1 = 0.9,
                           beta_2 = 0.999,
                           epsilon = 1,
                           write_output = False,
                           write_output_string = "output.csv",
                           fix_mass_adjustment = True,
                           fix_mass_target = 'initial',
                           fix_mass_round_value = 5,
                           initialize_first_and_second_vectors_from_sdf = False,
                           initialize_first_and_second_vector_target_sdf_file = "default",
                           initialize_first_and_second_vector_target_sdf_betas = [],
                           materials = ["void", "fuel/moderator:25/75"]):
    steps = 1
     
    if fix_mass_adjustment:
        if fix_mass_target == 'initial':
            target_mass  = sum(initial_betas)
        else:
            target_mass = fix_mass_target
    
    if initialize_first_and_second_vectors_from_sdf:
        assert initialize_first_and_second_vector_target_sdf_file != "default", "No target sdf specified for initial 1st, 2nd vectors."
        assert initialize_first_and_second_vector_target_sdf_betas != [], "No target sdf beta values specified."
        
        keff, beta_sensitivities, material_1_sense, material_2_sense = evaluate_with_Tsunami(material_betas,
                          materials,
                          tsunami_job_flag = initialize_first_and_second_vector_target_sdf_file,
                          build_input = False,
                          submit_tsunami_job = False,
                          pull_keff = False,
                          pull_sensitivities = True)
        
        first_moment_vector = [0] * 121
        second_moment_vector = [0] * 121
        first_moment_vector = calculate_first_moment_vector(beta_1, first_moment_vector, beta_sensitivities)
        second_moment_vector = calculate_second_moment_vector(beta_2, second_moment_vector, beta_sensitivities)
  
    else:   
        first_moment_vector = [0] * 121
        second_moment_vector = [0] * 121
    
    ### Stopping_criteria not implemented currently, only # of steps
    stopping_criteria = False
    variables = initial_betas
    
    if write_output:
        with open(write_output_string, 'w') as output_file:
            output_file.write("step, keff, step_size, old_betas, new_betas, negative_sense, beta_sensitivities\n")
    


    ### Main loop
    while steps < number_of_steps + 1 and stopping_criteria == False:
        print("Step #:", steps)
        ### 
        tsunami_job_flag = 'tsunami_job_' + str(steps)
        
        ### Evaluate with TSUNAMI
        keff, beta_sensitivities, material_1_sense, material_2_sense = evaluate_with_Tsunami(variables,
                                                             tsunami_job_flag = tsunami_job_flag,                    
                                                             submit_tsunami_job = submit_tsunami_job,
                                                             materials = materials)

            
        ### Mulitplying %keff derives by keff
        beta_sensitivities = [float(deriv * float(keff)) for deriv in beta_sensitivities]
        
        ### Implementation of ADAM gradient descent
        first_moment_vector = [(beta_1 * first_mv  + (1 - beta_1) * deriv) for first_mv, deriv in zip(first_moment_vector, beta_sensitivities)]
        second_moment_vector = [(beta_2 * second_mv + (1 - beta_2) * deriv**2) for second_mv, deriv in zip(second_moment_vector, beta_sensitivities)]
        first_moment_vector_hat = [(first_mv / (1 - beta_1**steps)) for first_mv in first_moment_vector]
        second_moment_vector_hat = [(second_mv/ (1 - beta_2**steps)) for second_mv in second_moment_vector]
        
            
        new_variables = [(beta + (alpha_value * first_mv) / (math.sqrt(second_mv) + epsilon)) for
                         beta, first_mv, second_mv in zip(variables, first_moment_vector_hat, second_moment_vector_hat)]

        if fix_mass_adjustment:    
            new_variables = fixed_mass_adjustment(new_variables,
                                                  target_mass,
                                                  debug=debug_print_all,
                                                  mass_round_dig = fix_mass_round_value)
        new_variables = np.array(new_variables)

        if debug_print_all:
            debug_write_out_11x11_list(beta_sensitivities, "beta_sensitivities")
            debug_write_out_11x11_list(first_moment_vector, "first_moment_vector")
            debug_write_out_11x11_list(second_moment_vector, "second_moment_vector")
            debug_write_out_11x11_list(first_moment_vector_hat, "first_moment_vector_hat")
            debug_write_out_11x11_list(second_moment_vector_hat, "second_moment_vector_hat")
            debug_write_out_11x11_list(variables, "variables")
            debug_write_out_11x11_list(new_variables, "new_variables_final")
        
        ### Writing out the output file
        if write_output:
            with open(write_output_string, 'a') as output_file:
                write_string = str(steps) + "," + str(keff)
                
                for _ in variables:
                    write_string += "," + str(_)
                
                for _ in new_variables:
                    write_string += "," + str(_)
                    
                for _ in beta_sensitivities:
                    write_string += "," + str(_)
                    
                for _ in material_1_sense:
                    write_string += "," + str(_)
                    
                for _ in material_2_sense:
                    write_string += "," + str(_)
                
                output_file.write(write_string + "\n")
        steps += 1        
        variables = new_variables


# In[9]:


### Building initial beta values
#material_betas = build_initial_betas(11, 11, 'random', rand_min = 0.2, rand_max = 0.8)
number_used=61
material_betas = build_initial_betas(11, 11, 'fixed', fixed_value = number_used/121)


### running adam gradient descent algo
adam_gradient_descent_scale(material_betas,
                       submit_tsunami_job=True,
                       debug_print_all = False,
                       alpha_value = 40.0,
                       number_of_steps = 10,
                       write_output = True,
                       fix_mass_adjustment = True,
                       fix_mass_target = 'initial',
                       fix_mass_round_value = 5,
                       materials = ["fuel/moderator:1/3.4", "void" ])

