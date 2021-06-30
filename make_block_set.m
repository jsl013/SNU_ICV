function [block_set, n_block] = make_block_set(win_size, min_block)
    step_size = [4, 6];
    n_scale = floor(win_size(1)-min_block(1))/4 + 1;
    scale_ratio = [1 1; 1 2; 2 1];
    block_set = [];
    for i=0:n_scale-1
        scale = 4*i + min_block(1);
        for j=1:3
            ratio = scale_ratio(j,:);
            w = ratio(1)*scale;
            h = ratio(2)*scale;
            if win_size(1) - w < 0 || win_size(2) - h < 0
                break;
            end
            for ss = step_size
                patch_x = floor((win_size(1)-w)/ss)+1;
                patch_y = floor((win_size(2)-h)/ss)+1;
                for x=0:patch_x-1
                    for y=0:patch_y-1
                        lefttop = [1+x*ss 1+y*ss];
                        wh = [w h];
                        block_set = [block_set ; lefttop wh]; % [x, y, w, h]
                    end
                end
            end
        end
    end
    
    block_set = unique(block_set, 'row');  % [x, y, w, h]
    n_block = size(block_set,1);
end