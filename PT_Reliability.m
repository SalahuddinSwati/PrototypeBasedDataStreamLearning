function [PT_Global_Density]=PT_Reliability()
    global dim;
    global Global_Meta;
    global levels;
    global global_labels;
    global PT_meta_all;
   % global gamma_all;
    PT_meta_last=PT_meta_all{levels};
    PT_Global_Density=zeros(size(PT_meta_last,1),1);
    for i=1:length(global_labels)
        label=global_labels(i);
        G_idx=cell2mat(Global_Meta(:,2))==label;
        G_meta=Global_Meta{G_idx,1};
        Aver=G_meta(1:dim)./G_meta(end);
        G_X=G_meta(end-1)./G_meta(end);
        idx_PT_cls=PT_meta_last(:,end)==label;
        temp_PT=PT_meta_last(idx_PT_cls,:);
        PT_centers=temp_PT(:,1:dim)./temp_PT(:,end-2);
        [L,W]=size(temp_PT);
        temp_GD_PT=1./(ones(L,1)+sum((PT_centers-repmat(Aver,L,1)).^2,2)./((G_X-sum(Aver.^2))));
        PT_Global_Density(idx_PT_cls)=temp_GD_PT;
    end
end