import os

def remove_consecutive_duplicates(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()

    temp_file_path = file_path + '.tmp'

    with open(temp_file_path, 'w', encoding='utf-8') as new_file:
        # Write the first four lines (headers, titles, etc.) as they are
        for line in lines[:4]:
            new_file.write(line)

        previous_line_data = None
        line_counter = 0  # Initialize line counter
        counter = 0
        # Start from the fifth line (index 4) to skip the first four lines
        for line in lines[4:]:
            line_data = line.strip().split('\t')

            if line_data[1:] != previous_line_data or counter == 4:
                # Update the t column to the current line number
                line_data[0] = str(line_counter)
                new_file.write('\t'.join(line_data) + '\n')
                line_counter += 1  # Increment line counter
                counter = 0
            if line_data[1:] == previous_line_data and counter < 4:
                counter += 1
            previous_line_data = line_data[1:]


    # Replace the original file with the new file
    os.replace(temp_file_path, file_path)

on_file = False
folder_path = 'data/validation_exp/frequency_analysis'
folder_path = 'data/used_reference_trajectories/Xing_trajectories/measures_interpolated'
file_name = 'data/exp10_01/exp12_01_Msolo_sinus_fast_20ms.txt'

if on_file : 
    remove_consecutive_duplicates(file_name)

else :
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            full_path = os.path.join(folder_path, file_name)
            remove_consecutive_duplicates(full_path)