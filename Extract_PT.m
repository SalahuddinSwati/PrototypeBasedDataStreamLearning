function [buffer]=Extract_PT(data)


train_data=data(:,1:end-1);
train_data_labels=train_data(:,end-1);
[train_cls_lb, ia2, cid2] = unique(train_data_labels);
buffer=[];
for lb=1:length(train_cls_lb)
    class_data=train_data(train_data_labels==train_cls_lb(lb),1:end-2);
     temp_class_data=class_data;
     
     G_mean=mean(class_data);
    G_mean_norm=norm(G_mean).^2;
    G_X=mean(sum(class_data.^2,2));
    diff_GX_G_mean=G_X-G_mean_norm;
  
    [UD,J,K]=unique(class_data,'rows');
    F = histc(K,1:numel(J));
    [L,W]=size(UD);
    Aver=G_mean;
    X=G_X;
    GlobalDensity=F./(ones(L,1)+sum((UD-repmat(Aver,L,1)).^2,2)./((X-sum(Aver.^2))));
    GlobalDensity=GlobalDensity(K,:);
    %[MM II]=sort(GlobalDensity);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     temp=mean(GlobalDensity)-std(GlobalDensity);
%     rem_idx=find(GlobalDensity<temp);
%     del_data=class_data(rem_idx,:);
%    
%     GlobalDensity(rem_idx)=[];
%     class_data(rem_idx,:)=[];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~, I]=max(GlobalDensity);
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
    
    
   % hold on;
   % scatter(class_data(u_star,1), class_data(u_star,2),20,'d','k');
    %hold off;
    
    membership=[];
    for i=1:size(class_data,1)
        
       Idx = knnsearch(class_data(u_star,:),class_data(i,:));
       membership=[membership; Idx];
       
    end
    Centers=[];
    N_C=[];
    for i=1:size(u_star,1)
        N_C=[N_C;sum(membership==i)];
        if sum(membership==i)==1
            Centers=[Centers; class_data(membership==i,:)];
        else
            Centers=[Centers; mean(class_data(membership==i,:))];
        end
            
    end
   %scatter(Centers(:,1), Centers(:,2),30,'d','g');
    
    D_C=[];
  for i=1:size(Centers,1)
        temp_norm=norm(Centers(i,:)-G_mean).^2;
        temp=N_C(i)/(1+(temp_norm/diff_GX_G_mean));
        D_C=[D_C;temp];
  end
  gamma_o=2*(diff_GX_G_mean);
  gamma(1)=sqrt(gamma_o)/2;
  gamma(2)=gamma_cal(class_data,gamma(1));
  gamma(3)=gamma_cal(class_data,gamma(2));
  gamma(4)=gamma_cal(class_data,gamma(3));
  
  i=4;

  temp_PT=PT_level(class_data, Centers, gamma_o,gamma(i-1),gamma(i),D_C,lb);
  for j=1:size(temp_PT,1)
      [idx, D]=knnsearch(temp_class_data,temp_PT(j,:),'NSMethod','exhaustive','k',1);
      temp_class_data(idx,:)=[];
  end
  buffer=[buffer; [temp_PT,ones(size(temp_PT,1),1).*train_cls_lb(lb), ones(size(temp_PT,1),1)]];
    buffer=[buffer; [temp_class_data,zeros(size(temp_class_data,1),1), zeros(size(temp_class_data,1),1)]];

end 
end
function [PT]=PT_level(class_data,Centers, gamma_o,gamma_p,gamma_c,D_C,lb)
  piar_D = pdist2(Centers,Centers);
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
    cost=ojective_fun(PT,class_data,gamma_o);
    for i=1:10
        [PT]=Voronoi_Clu(PT,class_data);
        cost=[cost;ojective_fun(PT,class_data,gamma_o)];
    end
    
    [PT]=Voronoi_Clu(PT,class_data);
    %%%%%%%%%%%%%%%%%%%%

end
function [cost]=ojective_fun(PT,class_data,gamma_o)
      pD_PT_X = pdist2(PT,class_data);
  pD_PT_X(pD_PT_X==0)=100;
  min_Dist_PT_X=min(pD_PT_X);
  J1=sum(min_Dist_PT_X)/(size(class_data,1)*gamma_o);
  cost=J1;
end
function [newPT]=Voronoi_Clu(PT,class_data)
      membership=[];
    for i=1:size(class_data,1)
        
       Idx = knnsearch(PT,class_data(i,:));
       membership=[membership; Idx];
       
    end
    Centers=[];
    N_C=[];
    for i=1:size(PT,1)
        N_C=[N_C;sum(membership==i)];
        N_C2=sum(membership==i);
        if sum(membership==i)==1
            Centers=[Centers; class_data(membership==i,:)];
        else
            Centers=[Centers; mean(class_data(membership==i,:))];
        end

    end

    newPT=Centers;

end
function []=drawcircle(x,y,r,sy)
   
    scatter(x, y,30,'b',sy);
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    plot(xunit, yunit);
end
function [gamma_val]=gamma_cal(X_data,gamma)
 piar_D = pdist2(X_data,X_data);
  piar_D(piar_D==0)=100;
 
  total_sum=0;
  N_h=0;
  for i=1:size(X_data,1)
       N_h=N_h+sum(piar_D(i,:)<=gamma);
       total_sum=total_sum+sum(piar_D(i,piar_D(i,:)<=gamma));
       
  end
  gamma_val=total_sum/N_h;
end