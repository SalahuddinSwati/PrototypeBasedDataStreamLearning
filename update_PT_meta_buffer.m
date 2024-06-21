function []=update_PT_meta_buffer(buffer,flg)
global dim;
global Global_Meta;
global levels;
global PT_meta_all;
global gamma_all;
global curTime;
PT_meta_last=PT_meta_all{levels};
[uni_cls_lb, ia2, cid2] = unique(buffer(:,end));
if flg==1
    for j=1:size(buffer,1)
        ins=buffer(j,1:dim);
        label=buffer(j,end);
        [idx,pre_label,D]=classify(ins);
        if label==pre_label
            PT_meta_last(idx,end-4)=PT_meta_last(idx,end-4)+1;
            PT_meta_last(idx,end-1)=curTime;
            
        else
            PT_meta_last(idx,end-4)=PT_meta_last(idx,end-4)-1;
        end
    end
    PT_meta_all{levels}=PT_meta_last;
end

updateTime();%update time

for i=1:length(uni_cls_lb)
    label=uni_cls_lb(i);
    
    data=buffer(buffer(:,end)==label,1:dim);
    N_data=size(data,1);
    LS=sum(data,1);
    LXsum=sum(sum(data.^2,2));
    G_idx=cell2mat(Global_Meta(:,2))==label;
    G_meta=Global_Meta{G_idx,1};
    old_class_mean=G_meta(1:dim)./G_meta(end);
    old_X_mean=G_meta(end-1)./G_meta(end);
    old_diff=old_X_mean-sum(old_class_mean.^2);
    %new mean and X
    G_meta(1:dim)=G_meta(1:dim)+LS;
    G_meta(end-1)=G_meta(end-1)+LXsum;
    G_meta(end)=G_meta(end)+N_data;
    
    new_class_mean=G_meta(1:dim)./G_meta(end);
    new_X_mean=G_meta(end-1)./G_meta(end);
    new_diff=new_X_mean-sum(new_class_mean.^2);
    
    Global_Meta{G_idx,1}=G_meta;
    
    gamma_idx=cell2mat(gamma_all(:,2))==label;
    gamma=gamma_all{gamma_idx,1};
    
    gamma(1)=2*(new_diff);
    gamma(2:end)=gamma(2:end).*(new_diff./old_diff);
    gamma(isnan(gamma))=0.00001;
    gamma(gamma<0.00001)=0.00001;
    gamma_all{gamma_idx,1}=gamma;
    
    
end
for k=1:size(buffer,1)
    ins=buffer(k,1:end-1);
    label=buffer(k,end);
    gamma_idx=find(cell2mat(gamma_all(:,2))==label);
    gamma=gamma_all{gamma_idx,1};
    for i=1:levels
        PT_H=PT_meta_all{i};
        nn_idx=find(PT_H(:,end)==label);
        LS=PT_H(nn_idx,1:dim);
        N=PT_H(nn_idx,end-2);
        centers=LS./N;
        LXsumPT=PT_H(nn_idx,end-3);
        Time=PT_H(nn_idx,end-1);
        RE=PT_H(nn_idx,end-4);
        [idx, D]=knnsearch(centers,ins,'NSMethod','exhaustive','k',1);
        D=D.^2;
 
        if D>gamma(i+1)

            pt_h_len=size(PT_H,1)+1;
            PT_H(pt_h_len,:)=[ins,1,sum(ins.^2),1,curTime,label];
            
            PT_meta_all{i}=PT_H;

        else
            LS(idx,:)=LS(idx,:)+ins;
            N(idx)=N(idx)+1;
            LXsumPT(idx)=LXsumPT(idx)+sum(ins.^2);
            Time(idx)=curTime;
            if RE(idx)<1
                RE(idx)=1;
            end
            PT_H(nn_idx,1:dim)=LS;
            PT_H(nn_idx,end-3)=LXsumPT;
            PT_H(nn_idx,end-2)=N;
            PT_H(nn_idx,end-1)=Time;
            PT_H(nn_idx,end-4)=RE;
            PT_meta_all{i}=PT_H;
            %gamma(i+1)=gamma(i+1)*(diff_GX_G_mean./old_diff);
        end
       
    end
    %     end
end

end