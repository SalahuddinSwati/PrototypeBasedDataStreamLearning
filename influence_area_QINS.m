function [final_q_idx,q_ins]=influence_area_QINS(buffer)
global dim;
global PT_meta_all;
global gamma_all;
global levels;

q_ins=[];
q_idx=[];
q_idx_last=[];

for j=1:size(buffer,1)
    ins=buffer(j,1:dim);
    pre_label=buffer(j,end-2);
    
    
    for i=1:levels
        PT_all=PT_meta_all{i};
        nn_idx=find(PT_all(:,end)~=pre_label);
        LS=PT_all(nn_idx,1:dim);
        N=PT_all(nn_idx,end-2);
        
        centers=LS./N;
        [idx, D]=knnsearch(centers,ins,'NSMethod','exhaustive','k',1);
        D=D.^2;
        if ~isempty(idx)
            nn_label=PT_all(nn_idx(idx),end);
            if length(nn_label)>1
                aaa=1;
            end
%             try
                gamma_idx=cell2mat(gamma_all(:,2))==nn_label;
%             catch
%                 aaa=1;
%             end
            
            gamma=gamma_all{gamma_idx,1};
            
            if D<gamma(i+1)
                
                
                if i==levels
                    q_idx_last=[q_idx_last;j];
                else
                    q_idx=[q_idx;j];
                    q_ins=[q_ins;ins];
                end
                break;
            end
        end
    end
end

membership=[];
PT_last=PT_meta_all{levels};
for i=1:size(q_ins,1)
    
    
    LS=PT_last(:,1:dim);
    N=PT_last(:,end-2);
    
    centers=LS./N;
    
    Idx = knnsearch(centers(:,:),q_ins(i,:));
    membership=[membership; Idx];
    
end
final_q_idx=[];
for i=1:size(PT_last,1)
    sum_count=sum(membership==i);
    if sum_count~=0
        
        temp_idx=q_idx(membership==i);
        final_q_idx=[final_q_idx;temp_idx(1)];
    end
    
end
final_q_idx=[final_q_idx;q_idx_last];
q_idx=[q_idx;q_idx_last];
end