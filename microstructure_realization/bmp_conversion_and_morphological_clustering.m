folder_name = 'Results/000888';
new_folder_name = [folder_name '_tertiary'];
[status, msg, msgID] = mkdir(new_folder_name);
number_of_slices = 180;
% im_3d = zeros(number_of_slices, number_of_slices, number_of_slices, 'logical');
im_3d = zeros(number_of_slices, number_of_slices, number_of_slices);

for i = 1:1:number_of_slices
    s = sprintf('%s/slice_%03d.png', folder_name, i);
    [im, map] = imread(s);
%     im = logical(im);
    im_3d(:, :, i) = im;
%     s = sprintf('%s/slice_%03d.png', new_folder_name, i);
%     imwrite(im_3d(:, :, i)+1, map, s);
end

%% String of two byte unsigned integers
lengths = 3 + 3 + number_of_slices^3;
micro_str = zeros(1, lengths, 'uint16');
nx = 180; ny = 180; nz = 180;
micro_str(1) = uint16(nx); micro_str(2) = uint16(ny); micro_str(3) = uint16(nz);
dx = 1; dy = 1; dz = 1;
micro_str(4) = uint16(dx); micro_str(5) = uint16(dy); micro_str(6) = uint16(dz);
for i = 1:1:number_of_slices
    for j = 1:1:number_of_slices
        for k = 1:1:number_of_slices
            micro_str(i+ (j-1)*nx + (k-1)*nx*ny + 6) = uint16(im_3d(i, j, k)); 
        end
    end
end
fileID = fopen('microstructure_string_unint16.bin', 'w');
fwrite(fileID, micro_str, 'uint16');

%% Morhological operations to convert the input microstructure into a fully clustered one
% 
% im_3d_morphed = zeros(number_of_slices, number_of_slices, number_of_slices, 'logical');
% structuring_element = strel('cube', 3);
% structuring_element = strel('sphere', 1);
% new_folder_name = [folder_name '_sphere1_eroded'];
% [status, msg, msgID] = mkdir(new_folder_name);
% 
% volume_fraction = 0;
% for i = 1:1:number_of_slices
%     for j = 1:1:number_of_slices
%         for k = 1:1:number_of_slices
%             if im_3d(i, j, k)
%                 volume_fraction = volume_fraction + 1;
%             end
%         end
%     end
% end
% volume_fraction_initial = volume_fraction / (number_of_slices^3);
% 
% iteration = 0;
% maximum_volume_fraction_cluster = 0;
% temp = 0;
% r = 1;
% im_3d_morphed = im_3d;
% while (maximum_volume_fraction_cluster < 0.95) && (volume_fraction / (number_of_slices^3) < 1.1 * volume_fraction_initial)
%     if maximum_volume_fraction_cluster <= temp
%         structuring_element = strel('sphere', r);
%         r = r+1;
%     else
%         r = 1;
%         structuring_element = strel('sphere', r);
%         temp = maximum_volume_fraction_cluster;
%     end
%     im_3d_morphed = imclose(im_3d_morphed, structuring_element);
%     %im_3d_morphed = imerode(im_3d_morphed, structuring_element);
%     connected_components = bwconncomp(im_3d_morphed, 6);
%     numPixels = cellfun(@numel, connected_components.PixelIdxList);
%     volume_fraction = 0;
%     for i = 1:1:number_of_slices
%         for j = 1:1:number_of_slices
%             for k = 1:1:number_of_slices
%                 if im_3d_morphed(i, j, k)
%                     volume_fraction = volume_fraction + 1;
%                 end
%             end
%         end
%     end
%     maximum_volume_fraction_cluster = max(numPixels) / volume_fraction;
%     iteration = iteration + 1;
% end
% im_3d_morphed = imfill(im_3d_morphed, 'holes');
% connected_components_background = bwconncomp(~im_3d_morphed, 6);
% numPixels_background = cellfun(@numel, connected_components_background.PixelIdxList);
% volume_fraction_background = 0;
% maximum_volume_fraction_cluster_background = 0;
% for i = 1:1:number_of_slices
%     for j = 1:1:number_of_slices
%         for k = 1:1:number_of_slices
%             if ~im_3d_morphed(i, j, k)
%                 volume_fraction_background = volume_fraction_background + 1;
%             end
%         end
%     end
% end
% maximum_volume_fraction_cluster_background = max(numPixels_background) / volume_fraction_background;
% 
% maximum_volume_fraction_cluster
% maximum_volume_fraction_cluster_background
% for i = 1:1:number_of_slices
%     s = sprintf('%s/slice_%d.png', new_folder_name, i);
%     imwrite(im_3d_morphed(:, :, i), s);
% end
