function []=class_PT_Extraction(data,b)
global total_fp;
global Global_Meta;
% global known_acc;
global levels;
global gamma_all;
global global_labels;
global train_cls_lb;
global PT_meta_all;
global novel_buff;
global novel_acc;
global Actual_Pred;
data=PT_Extraction(data);
if isempty(data)
    novel_buff=[];
else
    temp_label=data(:,end);
    Actual_Pred=[Actual_Pred;[temp_label,-1*ones(size(data,1),1)]];
    %%%%%%existing instances
    n_idx=~ismember(temp_label,global_labels);
    
    other_data=data(~n_idx,:);
    total_fp=total_fp+sum(~n_idx);
    update_PT_meta_buffer(other_data,1);%update with existing instances
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %novel instances;
    novel_buff=[];
 
    novel_buff=[novel_buff;[data(n_idx,:),ones(size(data(n_idx,:),1),1).*b]];
   
    train_data=data(n_idx,1:end-1);
    train_data_labels=data(n_idx,end);
    if size(train_data,1)>=200
        [train_cls_lb, ia2, cid2] = unique(train_data_labels,'stable');
        global_labels=[global_labels;train_cls_lb];
        novel_acc=novel_acc+sum(n_idx);
        for lb=1:length(train_cls_lb)
            class_data=train_data(train_data_labels==train_cls_lb(lb),:);
            
            D_size=size(class_data,1);
            gm_len=size(Global_Meta,1)+1;
            G_meta=[sum(class_data,1),sum(sum(class_data.^2,2)),D_size];
            Global_Meta{gm_len,1}=G_meta;
            Global_Meta{gm_len,2}=train_cls_lb(lb);
            Global_Meta{gm_len,3}=1;
            G_mean=mean(class_data);
            G_mean_norm=norm(G_mean).^2;
            G_X=mean(sum(class_data.^2,2));
            diff_GX_G_mean=abs(G_X-G_mean_norm);
            
            
            [UD,J,K]=unique(class_data,'rows');
            F = histc(K,1:numel(J));
            [L,W]=size(UD);
            Aver=G_mean;
            X=G_X;
            GlobalDensity=F./(ones(L,1)+sum((UD-repmat(Aver,L,1)).^2,2)./((abs(X-sum(Aver.^2)))));
            GlobalDensity=GlobalDensity(K,:);
            
            [M I]=max(GlobalDensity);
            data_idx=[1:size(GlobalDensity,1)]';
            data_idx(I)=[];
            R=I;
            for i=2:size(data_idx,1)
                Idx = knnsearch(class_data(data_idx,:),class_data(R(i-1),:));
                R=[R;data_idx(Idx)];
                data_idx(Idx)=[];
            end
            u_star=R(1);
            for i=2:size(R,1)-1
                if GlobalDensity(i)>GlobalDensity(i-1) && GlobalDensity(i)>GlobalDensity(i+1)
                    u_star=[u_star;R(i)];
                end
            end
            
            
            membership=[];
            for i=1:size(class_data,1)
                
                Idx = knnsearch(class_data(u_star,:),class_data(i,:));
                membership=[membership; Idx];
                
            end
            Centers=[];
            
            N_C=[];
            for i=1:size(u_star,1)
                sum_count=sum(membership==i);
                if sum_count~=0
                    N_C=[N_C;sum_count];
                    
                    if sum(membership==i)==1
                        Centers=[Centers; class_data(membership==i,:)];
                    else
                        Centers=[Centers; mean(class_data(membership==i,:))];
                    end
                end
                
            end
            
            D_C=[];
            for i=1:size(Centers,1)
                temp_norm=norm(Centers(i,:)-G_mean).^2;
                temp=N_C(i)/(1+(temp_norm/diff_GX_G_mean));
                D_C=[D_C;temp];
            end
            
            gamma(1)=2*(diff_GX_G_mean);
            
            for lv=2:levels+1
                gamma(lv)=gamma_cal(class_data,gamma(lv-1));
                
            end
            g_len=size(gamma_all,1)+1;
             gamma(isnan(gamma))=0.00001;
            gamma(gamma<0.00001)=0.00001;
            gamma_all{g_len,1}=gamma;
            gamma_all{g_len,2}=train_cls_lb(lb);
            for i=2:levels+1
                
                [PT, PT_Meta]=PT_level(class_data, Centers, gamma(1),gamma(i),D_C,lb);
                PT_meta_all{i-1}=[PT_meta_all{i-1};PT_Meta];
                
                
            end% level/h loop
            
        end% end of loop
        novel_buff=[];
    end % end of if
end
end% end of function
function [PT,PT_Meta]=PT_level(class_data,Centers, gamma_o,gamma_c,D_C,lb)
piar_D = pdist2(Centers,Centers,'squaredeuclidean');
piar_D(piar_D==0)=100;
nbr_clu={};
for i=1:size(Centers,1)
    idx=find(piar_D(i,:)<=gamma_c);
    nbr_clu{i}=idx;
end
PT=[];
for i=1:size(Centers,1)
    nbr_idx=nbr_clu{i};
    if ~isempty(nbr_idx)
        if D_C(i)>max(D_C(nbr_idx))
            PT=[PT; Centers(i,:)];
        end
    else
        PT=[PT; Centers(i,:)];
    end
end

%optimize PT
% cost=ojective_fun(PT,class_data,gamma_o);
% i=1;
% while 1
%     [PT, PT_Meta]=Voronoi_Clu(PT,class_data,lb);
%     cost=[cost;ojective_fun(PT,class_data,gamma_o)];
%     i=i+1;
%     if abs(cost(i)-cost(i-1))<0.0001
%         break;
%     end
% end

[PT, PT_Meta]=Voronoi_Clu(PT,class_data,lb);
%%%%%%%%%%%%%%%%%%%%

end
function [cost]=ojective_fun(PT,class_data,gamma_o)
pD_PT_X = pdist2(PT,class_data,'squaredeuclidean');
pD_PT_X(pD_PT_X==0)=100;
min_Dist_PT_X=min(pD_PT_X);
J1=sum(min_Dist_PT_X)/(size(class_data,1)*gamma_o);
cost=J1;
end
function [newPT, PT_Meta]=Voronoi_Clu(PT,class_data,lb)
global train_cls_lb;
global curTime;
membership=[];
for i=1:size(class_data,1)
    
    Idx = knnsearch(PT,class_data(i,:));
    membership=[membership; Idx];
    
end
Centers=[];
N_C=[];
PT_Meta=[];
for i=1:size(PT,1)
    N_C=[N_C;sum(membership==i)];
    if sum(membership==i)~=0
        N_C2=sum(membership==i);
        
        if sum(membership==i)==1
            Centers=[Centers; class_data(membership==i,:)];
        else
            Centers=[Centers; mean(class_data(membership==i,:))];
        end
        LS=sum(class_data(membership==i,:),1);
        LXsum=sum(sum(class_data(membership==i,:).^2,2));
        
        PT_Meta=[PT_Meta;LS,1,LXsum,N_C2,curTime,train_cls_lb(lb)];
    end
end

newPT=Centers;

end

function [gamma_val]=gamma_cal(X_data,gamma)
piar_D = pdist2(X_data,X_data,'squaredeuclidean');

piar_D(piar_D==0)=100;

total_sum=0;
N_h=0;
for i=1:size(X_data,1)
    N_h=N_h+sum(piar_D(i,:)<=gamma);
    total_sum=total_sum+sum(piar_D(i,piar_D(i,:)<=gamma));
    
end
gamma_val=total_sum/N_h;
end