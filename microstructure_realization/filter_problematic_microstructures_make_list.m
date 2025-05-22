path_to_log_files = 'Results';
if ~isempty(dir(fullfile(path_to_log_files, 'log_thread_*.txt')))
    dir_outputs = dir(fullfile(path_to_log_files, 'log_thread_*.txt'));
    number_of_microstructures = 0;
    number_of_microstructures_old = 0;
    microstructure_ids_index = 1;
    for i = 1:length(dir_outputs)
        log_file_path = fullfile(dir_outputs(i).folder, dir_outputs(i).name);
        file_id = fopen(log_file_path, 'r');
        while ~feof(file_id)
            first_line = fgets(file_id);
            extracted_data = textscan(first_line, '%d- End: time = %d:%*d:%*d\n');
            if ~isempty(extracted_data{1}) && ~isempty(extracted_data{2})
				if extracted_data{1} != microstructure_ids(number_of_microstructures)
					number_of_microstructures = number_of_microstructures + 1;
					microstructure_ids(number_of_microstructures) = extracted_data{1};
				end
			else
				extracted_data = textscan(first_line, '%d- Done: time = %d:%*d:%*d\n');
				if ~isempty(extracted_data{1}) && ~isempty(extracted_data{2})
					if extracted_data{1} != microstructure_ids(number_of_microstructures)
						number_of_microstructures = number_of_microstructures + 1;
						microstructure_ids(number_of_microstructures) = extracted_data{1};
					end
				end
			end
        end
        frewind(file_id);
        if number_of_microstructures > number_of_microstructures_old
            number_of_microstructures_old = number_of_microstructures;
            while ~feof(file_id)
                first_line = fgets(file_id);
                position = ftell(file_id);
                second_line = fgets(file_id);
                if second_line == -1
                    break
                end
                fseek(file_id, position, 'bof');
                extracted_data = textscan([first_line second_line], '%d- Phase  %*d: Volume Fraction(VF) = %f; Realization VF = %f; stagnant_iteration = %*d\n%*d- Microstructure evolution finished after   %*d iterations with   %*d seeds added in the process.\n');
                if ~isempty(extracted_data{1}) && ~isempty(extracted_data{2}) && ~isempty(extracted_data{3})
                    if microstructure_ids_index <= length(microstructure_ids) && extracted_data{1} == microstructure_ids(microstructure_ids_index)
                        if extracted_data{2} - 0.35 > extracted_data{3}
                            microstructure_ids(microstructure_ids_index) = [];
                            microstructure_ids_index = microstructure_ids_index - 1;
                            number_of_microstructures = number_of_microstructures - 1;
                            number_of_microstructures_old = number_of_microstructures_old - 1;
                        end
                        microstructure_ids_index = microstructure_ids_index + 1;
                    end
                end
            end
        end
        fclose(file_id);
    end
end

% Make a list of all volumes/microstructures
% The first column:
%   The literal name of the MATLAB data file (.mat)
%   containing the phase matrix of microstructure throughout the degradation
%   time (normally 36 months after and the initial microstructure state)
outputs_path = fullfile('..', '..', '..', 'ML', 'TransUNet', 'lists', 'lists_Degradation');
file_id_output = fopen(fullfile(outputs_path, 'all.lst'), 'w');
for microstructure_ids_index = 1:number_of_microstructures
    fprintf(file_id_output, 'StepMAT_%06d.mat\n', microstructure_ids(microstructure_ids_index));
end
fclose(file_id_output);

% Make a list of training volumes/microstructures
% The first column:
%   The literal name of the MATLAB data file (.mat)
%   containing the phase matrix of microstructure throughout the degradation
%   time (normally 36 months after and the initial microstructure state)
% The second column:
%   Degradation month (1 means the initial/0th state)
file_id_output = fopen(fullfile(outputs_path, 'train.txt'), 'w');
for microstructure_ids_index = 1:floor(number_of_microstructures*.8)
    for month = 1:1:36
        fprintf(file_id_output, 'StepMAT_%06d.mat %2d\n', microstructure_ids(microstructure_ids_index), month);
    end
end
fclose(file_id_output);

% Make a list of test volumes/microstructures
% The first column:
%   The literal name of the MATLAB data file (.mat)
%   containing the phase matrix of microstructure throughout the degradation
%   time (normally 36 months after and the initial microstructure state)
% The second column:
%   Degradation month (1 means the initial/0th state)
file_id_output = fopen(fullfile(outputs_path, 'test_vol.txt'), 'w');
for microstructure_ids_index = floor(number_of_microstructures*.8)+1:number_of_microstructures
    for month = 1:1:36
        fprintf(file_id_output, 'StepMAT_%06d.mat %2d\n', microstructure_ids(microstructure_ids_index), month);
    end
end
fclose(file_id_output);
