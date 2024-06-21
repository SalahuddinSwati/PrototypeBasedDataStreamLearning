function [f]=queury_ins_selection(pre_label,ins,PT_Global_Density)

    f=0;
    global PT_meta_all;
    global dim;
    global Global_Meta;
    global levels;
    PT_meta_last=PT_meta_all{levels};
    G_idx=cell2mat(Global_Meta(:,2))==pre_label;
    G_meta=Global_Meta{G_idx,1};
    Aver=G_meta(1:dim)./G_meta(end);
    G_X=G_meta(end-1)./G_meta(end);
    temp_GD_ins=1./(1+sum((ins-Aver).^2,2)./((G_X-sum(Aver.^2))));

    idx_PT_cls=PT_meta_last(:,end)==pre_label;
    if temp_GD_ins<(mean(PT_Global_Density(idx_PT_cls)))%-std(PT_Global_Density(idx_PT_cls)))
        f=1;
    end

end