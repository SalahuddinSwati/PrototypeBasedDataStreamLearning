function [f]=detect_novel(c_ins,PT_meta_all)
global dim;

global levels;
global gamma_all;
PT_meta_last=PT_meta_all{end};
centers=PT_meta_last(:,1:dim)./PT_meta_last(:,end-2);
[idx, D]=knnsearch(centers,c_ins,'NSMethod','exhaustive','k',1);
nn_label=PT_meta_last(idx,end);
gamma_idx=find(cell2mat(gamma_all(:,2))==nn_label);
gamma=gamma_all{gamma_idx,1};
f=-1;
for i=1:levels

    PT_meta_temp=PT_meta_all{i};
    idx_temp=find(PT_meta_temp(:,end)==nn_label);
    centers=PT_meta_temp(idx_temp,1:dim)./PT_meta_temp(idx_temp,end-2);
    
    [idx, D]=knnsearch(centers,c_ins,'NSMethod','exhaustive','k',1);
    dist=sum((centers(idx,:)-c_ins).^2);
    D=D.^2;
    if dist<gamma(i+1)
        f=0;
        break;
    end 
    
end