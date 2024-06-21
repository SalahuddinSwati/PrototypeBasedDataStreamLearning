function []=updateTime()
global PT_meta_all;
global levels;
global curTime;
global lambda;
global delT;
for i=1:levels
    PT=PT_meta_all{i};
    PT_time=PT(:,end-1);
    PT_Re_val=PT(:,end-4);
    Time_Cal=PT_Re_val.*exp(-(curTime-PT_time).*lambda);
    PT(:,end-4)=Time_Cal;
    idx=Time_Cal<=delT;
    if i==levels
        del_PT_data=PT(idx,:);
    end
    PT(idx,:)=[];
    PT_meta_all{i}=PT;


end
    if ~isempty(del_PT_data) && i==levels
        PT_meta_all= del_PT(del_PT_data,PT_meta_all,i);
    end
end
function [PT_meta_all]=del_PT(PT,PT_meta_all,l_idx)

global levels;
global gamma_all;
global Global_Meta;
global dim;
[uni_cls_lb, ia2, cid2] = unique(PT(:,end));
for i=1:length(uni_cls_lb)
    label=uni_cls_lb(i);
    idx=PT(:,end)==label;
    LS=PT(idx,1:dim);
    LS_Sum=sum(LS,1);
    LXsum=PT(idx,end-3);
    LXsum_Sum=sum(LXsum);
    N=PT(idx,end-2);
    N_Sum=sum(N);
    DelPT_center=LS./N;
    
    G_idx=cell2mat(Global_Meta(:,2))==label;
    G_meta=Global_Meta{G_idx,1};
    old_class_mean=G_meta(1:dim)./G_meta(end);
    old_X_mean=G_meta(end-1)./G_meta(end);
    old_diff=old_X_mean-sum(old_class_mean.^2);
    %new mean and X
    G_meta(1:dim)=G_meta(1:dim)-LS_Sum;
    G_meta(end-1)=G_meta(end-1)-LXsum_Sum;
    G_meta(end)=G_meta(end)-N_Sum;
    
    new_class_mean=G_meta(1:dim)./G_meta(end);
    new_X_mean=G_meta(end-1)./G_meta(end);
    new_diff=new_X_mean-sum(new_class_mean.^2);
    
    Global_Meta{G_idx,1}=G_meta;
    
    gamma_idx=cell2mat(gamma_all(:,2))==label;
    gamma=gamma_all{gamma_idx,1};
    gamma(1)=2*(new_diff);
    
    gamma(2:end)=gamma(2:end).*(new_diff./old_diff);
    gamma_all{gamma_idx,1}=gamma;
    for j=1:size(DelPT_center,1)
        for k=1:l_idx-1
            PT_H=PT_meta_all{k};

            nn_idx=find(PT_H(:,end)==label);
            LS2=PT_H(nn_idx,1:dim);
            N2=PT_H(nn_idx,end-2);
            centers=LS2./N2;
            LXsumPT=PT_H(nn_idx,end-3);
            [idx2, D2]=knnsearch(centers,DelPT_center(j,:),'NSMethod','exhaustive','k',1);
            D2=D2.^2;
            if D2<gamma(k+1)
                if N2(idx2)>N(j,:)
                    LS2(idx2,:)=LS2(idx2,:)-LS(j,:);
                    N2(idx2)=N2(idx2)-N(j,:);
                    LXsumPT(idx2)=LXsumPT(idx2)-LXsum(j);
                    PT_H(nn_idx,1:dim)=LS2;
                    PT_H(nn_idx,end-3)=LXsumPT;
                    PT_H(nn_idx,end-2)=N2;
                else
                    PT_H(nn_idx(idx2),:)=[];
                end
                PT_meta_all{k}=PT_H;
            end
            
        end
        
    end
end
end
