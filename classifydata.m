function[] =classifydata(edata)
global Actual_Pred;

global dim;
global known_acc;
global total_queury_ins;

global global_labels;

for id=1:size(edata,1)
    c_ins=edata(id,1:dim);
    c_ins_label=edata(id,end);
    [p_idx,pre_label,Dist]=classify(c_ins);
    Actual_Pred=[Actual_Pred;[c_ins_label,pre_label]];
    
    %             buffer=[buffer; [c_ins,pre_label,c_ins_label,0]];%0 is flg
    
    if c_ins_label==pre_label
        known_acc=known_acc+1;
        %             b_acc=b_acc+1;
    end
end
total_queury_ins=total_queury_ins+size(edata,1);
temp_label=edata(:,end);
n_idx=~ismember(temp_label,global_labels);
% novel_buff=[novel_buff;actual_buff(n_idx,:)];

edata=edata(~n_idx,:);
update_PT_meta_buffer(edata,1);%update with existing instances
end