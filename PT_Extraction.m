function [data]=PT_Extraction(data)

global Actual_Pred;
global gamma_all;
global dim;
global known_acc;
global total_queury_ins;
global levels;
global PT_meta_all;
global global_labels;
global novel_buff_size;
PT_last_ex=PT_meta_all{levels};

temp_gamma=cell2mat(gamma_all(:,1));
gamma_1=mean(temp_gamma(:,2));
gamma_0=mean(temp_gamma(:,1));
gamma_last=mean(temp_gamma(:,levels+1));
class_data=data(:,1:dim);

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
% while 1
D_C=[];
for i=1:size(Centers,1)
    temp_norm=norm(Centers(i,:)-G_mean).^2;
    temp=N_C(i)/(1+(temp_norm/diff_GX_G_mean));
    D_C=[D_C;temp];
end
% figure,
%
% scatter(data(:,1), data(:,2),20,'*','b');
% hold on

% dist00=pdist(Centers,'euclidean').^2;
% mdist01=mean(dist00);
%
% mdist02=mean(dist00(dist00<mdist01));
%
% mdist03=mean(dist00(dist00<mdist02));


[PT, PT_Count,center_labels,ins_idx_0]=PT_level(class_data, Centers, gamma_0,gamma_1,D_C,data);
ex_idx=[];
novel_idx=[];

if max(PT_Count)<size(data,1)/4
    classifydata(data);
    data=[];
else
    %     max(PT_Count);
    [PT_last, PT_Count_last,center_labels_last,ins_idx_last]=PT_level(class_data, Centers, gamma_last,gamma_1,D_C,data);
        PT_idx_low=PT_Count<200;
    
        AllC_idx = cell2mat({cat(1, ins_idx_0{PT_idx_low})});
        outPT_idx=[];
        novel_idx=cell2mat({cat(1, ins_idx_0{~PT_idx_low})});
        for pti=1:length(ins_idx_last)
            iddx=ins_idx_last{pti};
            if sum(ismember(iddx,AllC_idx))>0
                outPT_idx=[outPT_idx,pti];
            end
    
        end
    
    for pti=1:size(outPT_idx,1)
        [Nidx, ND]=knnsearch(PT_last_ex(:,1:dim),PT_last(outPT_idx(pti),:),'NSMethod','exhaustive','k',3);
        [Nidx1, ND2]=knnsearch(PT_last(:,:),PT_last(outPT_idx(pti),:),'NSMethod','exhaustive','k',4);
        Edis=mean(ND);
        Ndis=mean(ND2(2:end));
        EDP=Edis/(Edis+Ndis);
        NDP=Ndis/(Edis+Ndis);
        if NDP<EDP
            %f=[f;novel_buff(i,:)];
            novel_idx=[novel_idx;ins_idx_last{outPT_idx(pti)}];
            %             i=i+1;
        else
            %             ex_buff=[ex_buff;novel_buff(i,:)];
            %             novel_buff(i,:)=[];
            ex_idx=[ex_idx;ins_idx_last{outPT_idx(pti)}];
        end
        
    end
end

edata=data(ex_idx,:);
data(ex_idx,:)=[];
% if size(data,1)<novel_buff_size/2
%     edata=[edata;data];
%     data=[];
% end
if ~isempty(edata)
    classifydata(edata);
end

max(PT_Count)
%     Centers=PT;

% end
end% end of function
function [PT,PT_Count,center_labels,ins_idx]=PT_level(class_data,Centers, gamma_o,gamma_c,D_C,data)
%
% dist00=pdist(Centers,'euclidean').^2;
% mdist01=mean(dist00);
%
% mdist02=mean(dist00(dist00<mdist01));
%
% mdist03=mean(dist00(dist00<mdist02));

% [gamma_val]=gamma_cal(Centers,mdist02);

piar_D = pdist2(Centers,Centers,'squaredeuclidean');

piar_D(piar_D==0)=100;
nbr_clu={};
for i=1:size(Centers,1)
    idx=find(piar_D(i,:)<=gamma_o);
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


[PT, PT_Count,center_labels,ins_idx]=Voronoi_Clu(PT,class_data,data);
%%%%%%%%%%%%%%%%%%%%


end

function [newPT, PT_Count,center_labels,ins_idx]=Voronoi_Clu(PT,class_data,data)

membership=[];
for i=1:size(class_data,1)
    
    Idx = knnsearch(PT,class_data(i,:));
    membership=[membership; Idx];
    
end
Centers=[];

PT_Count=[];
center_labels={};
ins_idx={};
for i=1:size(PT,1)
    
    if  sum(membership==i)~=0
        N_C2=sum(membership==i);
        ins_idx{i}=find(membership==i);
        if sum(membership==i)==1
            Centers=[Centers; class_data(membership==i,:)];
        else
            Centers=[Centers; mean(class_data(membership==i,:))];
        end
        center_labels{i}=data(membership==i,end);
        %         scatter(data(membership==i,1), data(membership==i,2),30,'*');
        LS=sum(class_data(membership==i,:),1);
        LXsum=sum(sum(class_data(membership==i,:).^2,2));
        
        PT_Count=[PT_Count;N_C2];
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
