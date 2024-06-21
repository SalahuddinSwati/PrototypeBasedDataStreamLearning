function [idx,pre_label,D]=classify(c_ins)
%global Class_HPT;
global dim;
global levels;
global PT_meta_all;
PT_meta_last=PT_meta_all{levels};
centers=PT_meta_last(:,1:dim)./PT_meta_last(:,end-2);
[idx, D]=knnsearch(centers,c_ins,'NSMethod','exhaustive','k',1);
pre_label=PT_meta_last(idx,end);
end