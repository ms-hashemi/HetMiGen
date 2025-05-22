%% The program inputs are the folder location of image results, "folder_name", 
% and the resolution of images, "number_of_slices", and the padding, "pad", 
% considered during the realization (if no padding, then pad = 0) 
% 
% The outputs will be the binarized images according to the different foreground
% phases which are present in the realized microstrcuture.

% Preprocessing
folder_name = '150pixel_10threads/Results/000352';
number_of_slices = 150;
pad = 15;
% The prefix of the directory name of the binarized images for each phase
new_folder_name = [folder_name '_binary'];
% The 3D matrix which contains all 2D images' information in a binary/logical way
im_3d = zeros(number_of_slices, number_of_slices, number_of_slices, 'logical');

% Repeating the process for different foreground phases in the input images
for phase = 1:1:15
    %% Extracting the 3D im_3d matrix based on the binarized images
    % according to the material "phase" label (background/black phase has label zero)
    for i = 1+pad:1:number_of_slices+pad
        s = sprintf('%s/slice_%03d.bmp', folder_name, i);
        [im, map] = imread(s);
%         im = logical(im);
        im = im(1+pad:number_of_slices+pad, 1+pad:number_of_slices+pad);
        im = (im == phase);
        im_3d(:, :, i-pad) = im; % Writing the i-th binarized image into the 3D matrix
    end
    % Checking if there is at least one voxel with label "phase"
    % The condition will be false when there is no more foreground phases
    % to account for.
    if isempty(im_3d(im_3d == true))
        break
    end
    disp('')
    disp('-------------------')
    disp(['The current foreground phase of the binarized microstrcuture: Phase_id = ', num2str(phase)])
    
    %% Finding the clusters of voxels based on the von Neumann neighborhood
	% for the initial state of the microstructure
    
	% 1. The current foreground of the binarized microstrcuture
    disp('Initial state:')
    connected_components = bwconncomp(im_3d, 6);
    stats = regionprops3(connected_components, 'BoundingBox');
    clsuters_widths = stats.BoundingBox(:, 4:6);
    % "numPixels" is the array containing the number of voxels of each
    % identified cluster in the 3D matrix
    numPixels = cellfun(@numel, connected_components.PixelIdxList);
    [biggest, idx] = max(numPixels);
    % Calculating the number of voxels for the current "phase"
    volume_fraction = 0;
    for i = 1:1:number_of_slices
        for j = 1:1:number_of_slices
            for k = 1:1:number_of_slices
                if im_3d(i, j, k)
                    volume_fraction = volume_fraction + 1;
                end
            end
        end
    end
    fprintf('Phase Volume Fraction = %.3f\nThe biggest cluster volume/Its Phase Volume = %.3f\nThe biggest cluster width in x, y, z directions = %d, %d, %d\n', volume_fraction/(number_of_slices^3), biggest/volume_fraction, clsuters_widths(idx, 1), clsuters_widths(idx, 2), clsuters_widths(idx, 3));
    % Plotting the bar chart for clusters size distribution
    cluster_size = max(stats.BoundingBox(:, 4:6), [], 2)';
    cluster_size = sort(cluster_size);
    bar_chart = [];
    entry = 1;
    counter = 1;
    bar_chart(entry, :) = [cluster_size(1), counter];
    for i = 1:1:length(cluster_size)-1
        if cluster_size(i+1) == cluster_size(i)
            counter = counter + 1;
            bar_chart(entry, :) = [cluster_size(i), counter];
        else
            entry = entry + 1;
            counter = 1;
            bar_chart(entry, :) = [cluster_size(i+1), counter];
        end
    end
    figure;
    bar(bar_chart(:, 1)', bar_chart(:, 2)');
    title(['Phase' num2str(phase) ' cluster size distribution']);
    xlabel('Max Dimension of Cluster (in one dimension)');
    ylabel('Number of Clusters');
    
    % 2. The current background of the binarized microstrcuture
	connected_components_background = bwconncomp(~im_3d, 6);
    stats = regionprops3(connected_components_background, 'BoundingBox');
    clsuters_widths = stats.BoundingBox(:, 4:6);
    % "numPixels" is the array containing the number of voxels of each
    % identified cluster in the 3D matrix
    numPixels_background = cellfun(@numel, connected_components_background.PixelIdxList);
    [biggest_background, idx_background] = max(numPixels_background);
    fprintf('Background Volume Fraction = %.3f\nThe biggest cluster volume/Background Volume = %.3f\nThe biggest cluster width in x, y, z directions = %d, %d, %d\n', 1-volume_fraction/(number_of_slices^3), biggest_background/((number_of_slices^3)-volume_fraction), clsuters_widths(idx_background, 1), clsuters_widths(idx_background, 2), clsuters_widths(idx_background, 3));
    % Plotting the bar chart for clusters size distribution
    cluster_size = max(stats.BoundingBox(:, 4:6), [], 2)';
    cluster_size = sort(cluster_size);
    bar_chart = [];
    entry = 1;
    counter = 1;
    bar_chart(entry, :) = [cluster_size(1), counter];
    for i = 1:1:length(cluster_size)-1
        if cluster_size(i+1) == cluster_size(i)
            counter = counter + 1;
            bar_chart(entry, :) = [cluster_size(i), counter];
        else
            entry = entry + 1;
            counter = 1;
            bar_chart(entry, :) = [cluster_size(i+1), counter];
        end
    end
    figure;
    bar(bar_chart(:, 1)', bar_chart(:, 2)');
    title(['Phase' num2str(phase) ' background cluster size distribution']);
    xlabel('Max Dimension of Cluster (in one dimension)');
    ylabel('Number of Clusters');

	%% Removing all foreground clusters except the biggest one
    
    disp('After removing all clusters except the biggest one:')
    for i = 1:1:length(numPixels)
        if i ~= idx
            im_3d(connected_components.PixelIdxList{i}) = 0;
        end
    end
    for i = 1:1:length(numPixels_background)
        if i ~= idx_background
            im_3d(connected_components_background.PixelIdxList{i}) = 1;
        end
    end
    
    % 1. The current foreground of the binarized microstrcuture
    % Calculating the number of voxels for the current "phase"
    volume_fraction = 0;
    for i = 1:1:number_of_slices
        for j = 1:1:number_of_slices
            for k = 1:1:number_of_slices
                if im_3d(i, j, k)
                    volume_fraction = volume_fraction + 1;
                end
            end
        end
    end
    connected_components = bwconncomp(im_3d, 6);
    stats = regionprops3(connected_components, 'BoundingBox');
    clsuters_widths = stats.BoundingBox(:, 4:6);
    % "numPixels" is the array containing the number of voxels of each
    % identified cluster in the 3D matrix
    numPixels = cellfun(@numel, connected_components.PixelIdxList);
    [biggest, idx] = max(numPixels);
    fprintf('Phase Volume Fraction = %.3f\nThe biggest cluster volume/Its Phase Volume = %.3f\nThe biggest cluster width in x, y, z directions = %d, %d, %d\n', volume_fraction/(number_of_slices^3), biggest/volume_fraction, clsuters_widths(idx, 1), clsuters_widths(idx, 2), clsuters_widths(idx, 3));
    
	% 2. The current background of the binarized microstrcuture
	connected_components_background = bwconncomp(~im_3d, 6);
    stats = regionprops3(connected_components_background, 'BoundingBox');
    clsuters_widths = stats.BoundingBox(:, 4:6);
    % "numPixels" is the array containing the number of voxels of each
    % identified cluster in the 3D matrix
    numPixels_background = cellfun(@numel, connected_components_background.PixelIdxList);
    [biggest_background, idx_background] = max(numPixels_background);
    fprintf('Background Volume Fraction = %.3f\nThe biggest cluster volume/Background Volume = %.3f\nThe biggest cluster width in x, y, z directions = %d, %d, %d\n', 1-volume_fraction/(number_of_slices^3), biggest_background/((number_of_slices^3)-volume_fraction), clsuters_widths(idx_background, 1), clsuters_widths(idx_background, 2), clsuters_widths(idx_background, 3));
        
    %% The results directory for the current "phase"
    
	new_folder_name2 = [new_folder_name '_phase' num2str(phase)];
    [status, msg, msgID] = mkdir(new_folder_name2);
    % Writing 2D images inside the 3D matrix to the above directory
    for i = 1:1:number_of_slices
        s = sprintf('%s/slice_%03d.png', new_folder_name2, i);
        imwrite(im_3d(:, :, i), s);
    end
end
