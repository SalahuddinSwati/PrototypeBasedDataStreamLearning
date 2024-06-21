function []=Initial_class_PT_Extraction(data)

global Global_Meta;

global levels;
global gamma_all;
global PT_meta_all;
train_data=data(:,1:end-1);
train_data_labels=data(:,end);
global train_cls_lb;
[train_cls_lb, ia2, cid2] = unique(train_data_labels);

for m=1:levels
    PT_meta_all{m}=[];
end

for lb=1:length(train_cls_lb)
    class_data=train_data(train_data_labels==train_cls_lb(lb),:);
    
    D_size=size(class_data,1);
    G_meta=[sum(class_data,1),sum(sum(class_data.^2,2)),D_size];
    Global_Meta{lb,1}=G_meta;
    Global_Meta{lb,2}=train_cls_lb(lb);
    Global_Meta{lb,3}=1;
    G_mean=mean(class_data);
    G_mean_norm=norm(G_mean).^2;
    G_X=mean(sum(class_data.^2,2));
    diff_GX_G_mean=G_X-G_mean_norm;
    
    
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
    gamma=[];
     gamma(1)=2*(diff_GX_G_mean);
     
        
        for lv=2:levels+1
            gamma(lv)=gamma_cal(class_data,gamma(lv-1));
            
        end
        
 
    %      dist00=pdist(class_data,'euclidean').^2;
    %      gamma2=[];
    %     for tt=1:levels+1
    %         dist00(dist00>mean(dist00))=[];
    %         gamma2(tt)=mean(dist00);
    %     end
   
    gamma(isnan(gamma))=0.00001;
    gamma(gamma<0.00001)=0.00001;
    gamma_all{lb,1}=gamma;
    gamma_all{lb,2}=train_cls_lb(lb);
    for i=2:levels+1
        
        [PT, PT_Meta]=PT_level(class_data, Centers, gamma(1),gamma(i),D_C,lb);
        PT_meta_all{i-1}=[PT_meta_all{i-1};PT_Meta];
        
        
    end
    
end

end
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
if isempty(PT)
    PT=Centers;
end
[PT, PT_Meta]=Voronoi_Clu(PT,class_data,lb);


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
global Initial_TD_size;
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
        
        PT_Meta=[PT_Meta;LS,1,LXsum,N_C2,Initial_TD_size,train_cls_lb(lb)];
    end
end

newPT=Centers;

end
function []=drawcircle(x,y,r,clr,sy)

scatter(x, y,30,clr,sy);
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
plot(xunit, yunit,clr);
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