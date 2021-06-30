function sample = block_sampling(block_set, n_sample)
    random = randsample(size(block_set,1), n_sample);
    sample = block_set(random,:);
end  