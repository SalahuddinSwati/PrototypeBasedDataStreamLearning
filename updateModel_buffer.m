function []=updateModel_buffer(buffer)
global total_queury_ins;
global novel_buff;

global global_labels;
global dim;
global PT_meta_all;
%[PT_Global_Density]=PT_Reliability();

% Rel_pre_idx=[];
% for j=1:size(buffer,1)
%     ins=buffer(j,1:dim);
%     pre_label=buffer(j,end-2);
%     [f]=queury_ins_selection(pre_label,ins,PT_Global_Density);
%     if f==0
% %         Rel_pre_buffer=[Rel_pre_buffer;[ins,pre_label]];
%         Rel_pre_idx=[Rel_pre_idx;j];
%     end
% end
[q_idx,q_ins]=influence_area_QINS(buffer);
buffer(q_idx,end)=1;

total_queury_ins=total_queury_ins+sum(buffer(:,end)==1);

 pre_buff_idx=buffer(:,end)==0;
 pre_buff=[buffer(pre_buff_idx,1:dim),buffer(pre_buff_idx,end-2)];
 
 
update_PT_meta_buffer(pre_buff,0);

  actual_buff_idx=buffer(:,end)==1;
 actual_buff=[buffer(actual_buff_idx,1:dim),buffer(actual_buff_idx,end-1)];
 
temp_label=actual_buff(:,end);
n_idx=~ismember(temp_label,global_labels);
% novel_buff=[novel_buff;actual_buff(n_idx,:)];

actual_buff=actual_buff(~n_idx,:);

update_PT_meta_buffer(actual_buff,1);
%  update_PT_meta(actual_buff);

end